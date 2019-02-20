#!/usr/bin/env python

import argparse
import h5py
import numpy as np
import sys
import re

args_parser = argparse.ArgumentParser()
args_parser.add_argument("input", type=str, metavar="edges")
args_parser.add_argument("--blocks", "-b", type=int, default=8)
args_parser.add_argument("--out", "-o", type=str, default="out.h5")

try:
    options = args_parser.parse_args()
except Exception as e:
    print(e)
    exit(-1)

num_blocks = options.blocks
in_file = h5py.File(options.input, "r")
out_file = h5py.File(options.out, "w")

out_edges = out_file.create_group("edges")
in_edges = in_file["edges"]

class Progress(object):
    def __init__(self, count, message=''):
        self._total = count
        self._count = 0
        print(message, end='')
        print("  0%", end='')
        sys.stdout.flush()

    def __iadd__(self, increment):
        self._count += increment
        print("\b\b\b\b%3d%%" % (self._count / float(self._total) * 100),
              end='')
        if self._count >= self._total:
            print()
        sys.stdout.flush()
        return self

class OutMapping:

    class Index:
        pass

    def __init__(self, population):

        self.has_groups = "edge_group_id" in population
        num_edges = population["edge_type_id"].shape[0]
        afferent_index = population["indices"]["target_to_source"]
        self.num_nodes = afferent_index["node_id_to_ranges"].shape[0]

        self.edge_mapping = np.zeros(num_edges, dtype="u8")
        self.edge_count = 0

        self.index = self.Index()
        # [[] * num_nodes] duplicates the same list, we need different lists
        self.index.afferent = []
        self.index.afferent.extend([[] for i in range(self.num_nodes)])
        self.index.efferent = []
        self.index.efferent.extend([[] for i in range(self.num_nodes)])

def copy_group_over(in_pop, out_pop, name, shuffling):
    out_group = out_pop.create_group(name)
    in_group = in_pop[name]
    for prop in in_group:
        out_data = out_group.create_dataset(
            prop, np.array(in_group[prop])[shuffling])

def make_efferent_view(edges):

    edges = np.vstack(edges)
    # Sorting the edges by source node (and then target node).
    # Using merge sort as it's guaranteed to be stable
    edges = edges[np.argsort(edges[:, 2], kind="mergesort"), :]
    # Swapping columns to edge id, source node, target node
    edges = edges[:, [0, 2, 1]]
    # Splitting the array in lists
    ids, counts = np.unique(edges[:, 1], return_counts = True)
    offsets = np.zeros(len(counts), dtype="u4")
    offsets[1:] = np.cumsum(counts[:-1])
    return [edges[offset:offset + count, :]
            for offset, count in zip(offsets, counts)]

def process_group_mapping(group_id, in_pop, out_mapping):

    num_nodes = out_mapping.num_nodes
    nodes_per_block = (num_nodes + num_blocks - 1) // num_blocks

    # Finding the mask to filter edges belonging to this group unless the
    # population is single group.
    if out_mapping.has_groups:
        group_ids = np.array(in_pop["edge_group_id"])
        mask = group_ids == group_id
    else:
        mask = None

    edge_id_mapping = out_mapping.edge_mapping
    edge_count = out_mapping.edge_count

    target_nodes = np.array(in_pop["target_node_id"], dtype="u8")
    source_nodes = np.array(in_pop["source_node_id"], dtype="u8")

    in_index = in_pop["indices"]["target_to_source"]
    node_to_ranges = in_index["node_id_to_ranges"]
    range_to_edges = in_index["range_to_edge_id"]

    afferent_index = out_mapping.index.afferent
    efferent_index = out_mapping.index.efferent

    progress = Progress(num_blocks * num_blocks,
                        "Processing group %d mapping: " % group_id)

    for row in range(0, num_nodes, nodes_per_block):

        edges = []

        # Getting all edge ids for nodes in range(i, i + nodes_per_block)
        ranges = node_to_ranges[row:row + nodes_per_block]
        for i, (a, b) in enumerate(ranges):
            if a == b:
                continue
            node_id = row + i
            # Checking that the edge layout is an afferent view (single range
            # per target neuron)
            assert(a + 1 == b)

            start, end = range_to_edges[a]

            # Getting the edge id range masked by the current group
            edge_ids = np.array(range(start, end), dtype="u8")
            if mask is not None:
                edge_ids = edge_ids[mask[start:end]]
            # And then the source and target ids
            target_ids = target_nodes[edge_ids]
            source_ids = source_nodes[edge_ids]
            # Checking that source_ids are sorted
            assert((np.diff(source_ids) >= 0).all())
            # and that all target ids are equal to the expected node_id
            assert((target_ids == node_id).all())

            edges.append(np.column_stack((edge_ids, target_ids, source_ids)))

        afferent_layout = row % 2 == 0

        for column in range(0, num_nodes, nodes_per_block):

            # Filtering edge ids by edges in this presynaptic block
            edge_subset = [
                edge_data[(edge_data[:, 2] >= column) &
                          (edge_data[:, 2] < (column + nodes_per_block))] for
                edge_data in edges]

            if afferent_layout:
                major_index = afferent_index
                minor_index = efferent_index
            else:
                major_index = efferent_index
                minor_index = afferent_index
                # Rebuilding edge_subsets from the afferent view
                edge_subset = make_efferent_view(edge_subset)

            for i, edge_data in enumerate(edge_subset):
                num_edges = edge_data.shape[0]
                if num_edges == 0:
                    continue

                # Dumping edge ids to the reoder mapping
                edge_id_mapping[edge_count:
                                edge_count + num_edges] = edge_data[:, 0]

                # Adding the edge ranges to the output indices
                # The major view is the easy one
                major_id = edge_data[0, 1]
                major_index[major_id].append((edge_count,
                                              edge_count + num_edges))
                # For the minor index we need to count how many edges are
                # coming/going from the same minor node.
                # Nodes are known to be sorted by the asserts in the loop
                # building the edges list
                ids, counts = np.unique(edge_data[:, 2], return_counts = True)
                for id, count in zip(ids, counts):
                    minor_index[id].append((edge_count, edge_count + count))
                    # The global edge count is advanced here as we need to
                    # update it for every efferent node
                    edge_count += count

            afferent_layout = not afferent_layout

            progress += 1

    out_mapping.edge_count = edge_count

def write_population_group_indices(out_pop, edge_mapping, group_sizes):

    group_ids = np.zeros(len(edge_mapping), dtype="u4")
    group_indices = np.zeros(len(edge_mapping), dtype="u8")
    offset = 0
    for group, size in group_sizes:
        group_ids[offset:offset + size] = group
        group_indices[offset:offset + size] = range(0, size)
        offset += size

    out_pop.create_dataset("edge_group_id", data=group_ids)
    out_pop.create_dataset("edge_group_index", data=group_indices)

def write_population_datasets(in_pop, out_pop, edge_mapping):

    for name in ["edge_type_id", "source_node_id", "target_node_id"]:
        out_pop.create_dataset(name, data=np.array(in_pop[name])[edge_mapping])

    for name in ["source_node_id", "target_node_id"]:
        out_pop[name].attrs["node_population"] = \
            in_pop[name].attrs["node_population"]

def write_index(population, mapping):

    def write_index_data(group, index):
        cum_range_count = np.cumsum([len(l) for l in index])
        nodes_to_ranges = np.zeros((len(index), 2), dtype="u8")
        nodes_to_ranges[1:, 0] = cum_range_count[:-1]
        nodes_to_ranges[:, 1] = cum_range_count
        group.create_dataset("node_id_to_ranges", data=nodes_to_ranges)
        # We need to filter empty lists before stacking
        index = [l for l in index if len(l) != 0]
        if len(index) == 0:
            group.create_dataset("range_to_edge_id", shape=(2, 0), dtype="u8")
            return
        ranges_to_edges = np.vstack(index)

        group.create_dataset("range_to_edge_id", data=ranges_to_edges)

    indices = population.create_group("indices")

    write_index_data(indices.create_group("target_to_source"),
                     mapping.index.afferent)
    write_index_data(indices.create_group("source_to_target"),
                     mapping.index.efferent)

def write_group(in_group, out_group, edge_mapping):

    for name in in_group:
        out_group.create_dataset(
            name, data=np.array(in_group[name])[edge_mapping])

def process_population(in_pop, out_pop):

    try:
        in_pop["edge_id"]
        print("Explicit edge IDs are not supported, skiping population")
        return
    except:
        pass

    out_mapping = OutMapping(in_pop)

    if out_mapping.has_groups:
        group_ids = np.array(in_pop["edge_group_id"])
        # Ensuring that edges from different groups are not mixed up in the
        # main index
        ids = np.array(group_ids)
        assert((np.diff(ids) >= 0).all())

        group_indices = np.array(in_pop["edge_group_index"])
        group_sizes = []

    for name in in_pop:
        if not re.match("[0-9]+", name):
            continue

        group_id = int(name)
        current_edge_count = out_mapping.edge_count

        process_group_mapping(group_id, in_pop, out_mapping)

        edge_mapping = out_mapping.edge_mapping

        if out_mapping.has_groups:
            # This code doesn't assume that the original group indices are sorted
            selection = slice(current_edge_count, out_mapping.edge_count)
            edge_mapping = group_indices[edge_mapping[selection]]

            group_size = out_mapping.edge_count - current_edge_count
            group_sizes.append((group_id, group_size))
        else:
            assert(out_mapping.edge_count == len(edge_mapping))

        write_group(in_pop[name], out_pop.create_group(name), edge_mapping)

    edge_mapping = out_mapping.edge_mapping
    assert(out_mapping.edge_count == len(edge_mapping))

    write_population_datasets(in_pop, out_pop, edge_mapping)
    if out_mapping.has_groups:
        write_population_group_indices(out_pop, edge_mapping, group_sizes)

    write_index(out_pop, out_mapping)


for population_name in in_edges:
    process_population(in_edges[population_name],
                       out_edges.create_group(population_name))





