import math
import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from chunkanon.categorical import CategoricalGeneralizer


class OLA_2:
    """
    OLA_2 class implements an optimized lattice anonymization strategy for chunk-based k-anonymity.
    It supports both categorical and numerical quasi-identifiers, and builds a generalization tree to find
    the best trade-off between data utility and privacy within a suppression threshold.
    """

    def __init__(self, quasi_identifiers, total_records, suppression_limit, multiplication_factors):
        self.quasi_identifiers = quasi_identifiers
        self.total_records = total_records
        self.suppression_limit = suppression_limit
        self.multiplication_factors = multiplication_factors
        self.tree = []
        self.smallest_passing_rf = None
        self.node_status = {}
        self.suppression_count = 0
        self.categorical_generalizer = CategoricalGeneralizer()
        self.domains = []

    def build_tree(self, initial_ri):
        self.tree = [[initial_ri]]
        if initial_ri is None:
            raise ValueError("initial_ri is None.")

        self.node_status = {tuple(initial_ri): None}

        while True:
            next_level = []
            for node in self.tree[-1]:
                for i in range(len(node)):
                    new_node = node.copy()
                    qi = self.quasi_identifiers[i]

                    if qi.is_categorical:
                        if qi.column_name == "Blood Group":
                            max_level = 3
                        elif qi.column_name == "Gender":
                            max_level = 2
                        else:
                            max_level = 4

                        if new_node[i] < max_level:
                            new_node[i] += 1
                            self._add_node_if_new(new_node, next_level)
                    else:
                        max_val = qi.get_range()
                        factor = self.multiplication_factors[
                            qi.column_name[:-8] if qi.is_encoded else qi.column_name
                        ]
                        if new_node[i] < max_val:
                            new_val = min(new_node[i] * factor, max_val)
                            new_node[i] = new_val
                            self._add_node_if_new(new_node, next_level)

            if not next_level:
                break
            self.tree.append(next_level)

        return self.tree

    def _add_node_if_new(self, node, level_list):
        """
        Add a new generalization node to the tree if it hasn't been added before.

        Parameters:
        - node (list): A generalization vector to add.
        - level_list (list): Current level of nodes to update.
        """
        t_node = tuple(node)
        if t_node not in self.node_status:
            level_list.append(node)
            self.node_status[t_node] = None

    def build_domains(self, dataset):
        self.domains = []
        for qi in self.quasi_identifiers:
            col = qi.column_name
            if qi.is_categorical:
                unique_values = dataset[col].unique()
                self.domains.append(sorted(unique_values.tolist()))
            else:
                self.domains.append(sorted(dataset[col].unique().tolist()))

    def get_index_tuple(self, row):
        indices = []
        for i, qi in enumerate(self.quasi_identifiers):
            val = row[qi.column_name]
            if qi.is_categorical:
                try:
                    index = self.domains[i].index(val)
                except ValueError:
                    raise ValueError(f"Value '{val}' not found in domain for {qi.column_name}")
            else:
                # Clamp the index to valid range
                index = min(np.searchsorted(self.domains[i], val), len(self.domains[i]) - 1)
            indices.append(index)
        return tuple(indices)


    def process_chunk(self, chunk, bin_widths):
        if not self.domains:
            self.build_domains(chunk)

        shape = [len(dom) for dom in self.domains]
        histogram = np.zeros(shape, dtype=int)

        for _, row in chunk.iterrows():
            idx = self.get_index_tuple(row)
            histogram[idx] += 1

        return histogram

    def merge_axis(self, array, axis, group_size):
        shape = list(array.shape)
        new_size = (shape[axis] + group_size - 1) // group_size
        pad = (new_size * group_size) - shape[axis]

        if pad:
            pad_shape = list(shape)
            pad_shape[axis] = pad
            padding = np.zeros(pad_shape, dtype=array.dtype)
            array = np.concatenate((array, padding), axis=axis)

        shape = array.shape
        reshaped = np.reshape(array, shape[:axis] + (new_size, group_size) + shape[axis+1:])

        return np.sum(reshaped, axis=axis+1)

    def merge_equivalence_classes(self, histogram, new_bin_widths):
        merged = histogram.copy()
        for i, (qi, bw) in enumerate(zip(self.quasi_identifiers, new_bin_widths)):
            if not qi.is_categorical:
                group_size = max(1, int(bw))
                merged = self.merge_axis(merged, i, group_size)
        return merged

    def check_k_anonymity(self, histogram, k):
        self.suppression_count = 0
        failing_mask = (histogram < k) & (histogram > 0)
        self.suppression_count = histogram[failing_mask].sum()
        return not failing_mask.any()

    def merge_histograms(self, histograms):
        return np.sum(np.array(histograms), axis=0)

    def get_final_binwidths(self, histogram, k):
        pass_nodes = []
        histogram_const = histogram.copy()
        total_nodes = sum(len(level) for level in self.tree)
        pbar = tqdm(total=total_nodes, desc="Marking Nodes", unit="node")

        while any(v is None for v in self.node_status.values()):
            histogram = histogram_const.copy()
            unmarked_levels = [i for i in range(len(self.tree)) if any(self.node_status.get(tuple(node)) is None for node in self.tree[i])]
            if not unmarked_levels:
                break

            mid_level = unmarked_levels[len(unmarked_levels) // 2]
            sorted_nodes = sorted(
                [node for node in self.tree[mid_level] if self.node_status[tuple(node)] is None], reverse=True
            )

            if sorted_nodes:
                node = sorted_nodes[len(sorted_nodes) // 2]
                histogram = self.merge_equivalence_classes(histogram, node)

                if self.node_status[tuple(node)] is not None:
                    continue

                if self.check_k_anonymity(histogram, k):
                    self._mark_subtree_pass(node, pbar)
                    pass_nodes.append(node)
                elif self.suppression_count < (self.suppression_limit * self.total_records):
                    self._mark_subtree_pass(node, pbar)
                    pass_nodes.append(node)
                else:
                    self._mark_parents_fail(node, pbar)

        pbar.close()
        self.find_best_rf(histogram_const, pass_nodes, k)
        return self.smallest_passing_rf if self.smallest_passing_rf else list(self.tree[0][0])

    def _mark_subtree_pass(self, node, pbar=None):
        q = [node]
        while q:
            current = q.pop(0)
            key = tuple(current)
            if self.node_status.get(key) is None:
                self.node_status[key] = 'pass'
                if pbar: pbar.update(1)

            for level in self.tree:
                for child in level:
                    t_child = tuple(child)
                    if all(child[i] >= current[i] for i in range(len(child))) and self.node_status.get(t_child) is None:
                        self.node_status[t_child] = 'pass'
                        q.append(child)
                        if pbar: pbar.update(1)

    def _mark_parents_fail(self, node, pbar=None):
        q = [node]
        while q:
            current = q.pop(0)
            key = tuple(current)
            if self.node_status.get(key) is None:
                self.node_status[key] = 'fail'
                if pbar: pbar.update(1)

            for level in reversed(self.tree):
                for parent in level:
                    parent_key = tuple(parent)
                    if all(parent[i] <= current[i] for i in range(len(current))) and self.node_status.get(parent_key) is None:
                        self.node_status[parent_key] = 'fail'
                        q.append(parent)
                        if pbar: pbar.update(1)

    def find_best_rf(self, histogram, pass_nodes, k):
        best_node = None
        lowest_dm_star = float('inf')
        print("\nPassing nodes and their DM* values:")

        for node in pass_nodes:
            merged_hist = self.merge_equivalence_classes(histogram.copy(), list(node))
            low_count_sum = np.sum(merged_hist[merged_hist < k])
            dm_star = np.sum(merged_hist[merged_hist >= k] ** 2) + low_count_sum ** 2

            print(f"Node: {list(node)}, DM*: {dm_star}")
            if dm_star < lowest_dm_star:
                lowest_dm_star = dm_star
                best_node = node

        self.smallest_passing_rf = best_node
        if best_node is not None:
            print(f"\nBest Node: {list(best_node)}, Final DM*: {lowest_dm_star}")
        else:
            print("No best node found.")

    def generalize_chunk(self, chunk, bin_widths):
        """
        Apply generalization to each record in a chunk based on given bin widths.

        Parameters:
        - chunk (DataFrame): Chunk to be generalized.
        - bin_widths (list): Generalization levels or bin widths for each QI.

        Returns:
        - DataFrame: Generalized chunk.
        """
        gen_chunk = chunk.copy(deep=False)

        for qi, bw in zip(self.quasi_identifiers, bin_widths):
            col = qi.column_name

            if qi.is_categorical:
                level = int(bw)
                mapping = {
                    "Blood Group": self.categorical_generalizer.generalize_blood_group,
                    "Gender": self.categorical_generalizer.generalize_gender
                }
                mapper = mapping.get(qi.column_name, self.categorical_generalizer.generalize_profession)

                gen_chunk[col] = gen_chunk[col].map(lambda x: mapper(x, level))

            else:
                col_data = gen_chunk[col]
                encoding_file = os.path.join("encodings", f"{col.replace(' ', '_').lower()}_encoding.json")

                if os.path.exists(encoding_file):
                    with open(encoding_file, "r") as f:
                        decoding_map = json.load(f)

                    bin_edges = np.arange(col_data.min(), col_data.max() + bw + 1, bw)
                    labels = [
                        f"[{decoding_map.get(int(bin_edges[i]), bin_edges[i])}-{decoding_map.get(int(bin_edges[i+1]-1), bin_edges[i+1]-1)}]"
                        for i in range(len(bin_edges) - 1)
                    ]

                    gen_chunk[col] = pd.cut(col_data, bins=bin_edges, labels=labels, include_lowest=True)

                else:
                    bin_edges = np.arange(qi.min_value, col_data.max() + bw, bw)
                    labels = [
                        f"[{int(bin_edges[i])}-{int(bin_edges[i + 1] - 1)}]"
                        for i in range(len(bin_edges) - 1)
                    ]

                    gen_chunk[col] = pd.cut(col_data, bins=bin_edges, labels=labels, include_lowest=True)

        return gen_chunk
    
    def combine_generalized_chunks_to_csv(self, generalized_chunks, output_path='generalized_chunk1.csv'):
        combined = pd.concat(generalized_chunks, ignore_index=True)
        combined.to_csv(output_path, index=False)
        print(f"Generalized data saved to {output_path}")
        return combined

    def get_suppressed_percent(self, node, histogram, k):
        histogram = self.merge_equivalence_classes(histogram, list(node))
        self.suppression_count = np.sum((histogram > 0) & (histogram < k))
        return (self.suppression_count / self.total_records) * 100
