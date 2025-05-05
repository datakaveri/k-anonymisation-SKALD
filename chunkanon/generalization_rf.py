import math
import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from chunkanon.categorical import CategoricalGeneralizer


class OLA_2:
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

    def build_tree(self, initial_ri):
        self.tree = [[initial_ri]]
        self.node_status = {tuple(initial_ri): None}

        while True:
            next_level = []
            for node in self.tree[-1]:
                for i in range(len(node)):
                    new_node = node.copy()

                    qi = self.quasi_identifiers[i]
                    if qi.is_categorical:
                        max_level = 3 if qi.column_name == "Blood Group" else 4
                        if new_node[i] < max_level:
                            new_node[i] += 1
                            self._add_node_if_new(new_node, next_level)
                    else:
                        max_val = qi.get_range()
                        if new_node[i] < max_val:
                            if (qi.is_encoded) :
                                factor = self.multiplication_factors[qi.column_name[:-8]]
                            else:
                                factor = self.multiplication_factors[qi.column_name]
                            new_val = min(new_node[i] * factor, max_val)
                            new_node[i] = new_val
                            self._add_node_if_new(new_node, next_level)

            if not next_level:
                break
            self.tree.append(next_level)

        return self.tree

    def _add_node_if_new(self, node, level_list):
        t_node = tuple(node)
        if t_node not in self.node_status:
            level_list.append(node)
            self.node_status[t_node] = None

    def process_chunk(self, chunk, bin_widths):
        equivalence_classes = {}

        for _, row in chunk.iterrows():
            key_parts = []
            for qi, bin_width in zip(self.quasi_identifiers, bin_widths):
                if qi.is_categorical:
                    level = int(bin_width)
                    if qi.column_name == 'Blood Group':
                        value = self.categorical_generalizer.generalize_blood_group(row[qi.column_name], level)
                    else:
                        value = self.categorical_generalizer.generalize_profession(row[qi.column_name], level)
                    key_parts.append(str(value))
                else:
                    val = row[qi.column_name]
                    start = qi.min_value + ((val - qi.min_value) // bin_width) * bin_width
                    end = start + bin_width - 1
                    key_parts.append(f"[{start}-{end}]")

            key = tuple(key_parts)
            equivalence_classes[key] = equivalence_classes.get(key, 0) + 1

        return equivalence_classes

    def merge_equivalence_classes(self, histogram, new_bin_widths):
        merged_histogram = {}

        def generalize(old_range, qi_index, new_bin_width):
            qi = self.quasi_identifiers[qi_index]
            if qi.is_categorical:
                if qi.column_name == 'Blood Group':
                    return self.categorical_generalizer.generalize_blood_group(old_range, int(new_bin_width))
                else:
                    return self.categorical_generalizer.generalize_profession(old_range, int(new_bin_width))
            else:
                min_val = float(old_range.split('-')[0].strip('['))
                start = qi.min_value + ((min_val - qi.min_value) // new_bin_width) * new_bin_width
                end = start + new_bin_width - 1
                return f"[{start}-{end}]"

        for eq_class, count in histogram.items():
            new_class = tuple(
                generalize(attr, i, bw)
                for i, (attr, bw) in enumerate(zip(eq_class, new_bin_widths))
            )
            merged_histogram[new_class] = merged_histogram.get(new_class, 0) + count

        return merged_histogram



    def generalize_chunk(self, chunk, bin_widths):
        gen_chunk = chunk.copy(deep=False)

        for qi, bw in zip(self.quasi_identifiers, bin_widths):
            col = qi.column_name

            if qi.is_categorical:
                level = int(bw)
                mapper = (
                    self.categorical_generalizer.generalize_blood_group if col == 'Blood Group'
                    else self.categorical_generalizer.generalize_profession
                )
                gen_chunk[col] = gen_chunk[col].map(lambda x: mapper(x, level))

            else:
                min_val = qi.min_value
                col_data = gen_chunk[col]

                
                encoding_file = os.path.join("encodings", f"{col.replace(' ', '_').lower()}_encoding.json")
                if os.path.exists(encoding_file):
                    
                    with open(encoding_file, "r") as f:
                        decoding_map = {k: v for k, v in json.load(f).items()}

                    bin_edges = np.arange(col_data.min(), col_data.max() + bw + 1, bw)

                    labels = []
                    for i in range(len(bin_edges) - 1):
                        low_enc = int(bin_edges[i])
                        high_enc = int(bin_edges[i + 1] - 1)

                        low_val = decoding_map.get(low_enc, low_enc)
                        high_val = decoding_map.get(high_enc, high_enc)

                        labels.append(f"[{low_val}-{high_val}]")

                    
                    gen_chunk[col:-8] = pd.cut(
                        col_data,
                        bins=bin_edges,
                        labels=labels,
                        include_lowest=True
                    )

                else:
                    
                    bin_edges = np.arange(min_val, col_data.max() + bw, bw)
                    labels = [
                        f"[{int(bin_edges[i])}-{int(bin_edges[i + 1] - 1)}]"
                        for i in range(len(bin_edges) - 1)
                    ]

                    gen_chunk[col] = pd.cut(
                        col_data,
                        bins=bin_edges,
                        labels=labels,
                        include_lowest=True
                    )

        return gen_chunk


    def check_k_anonymity(self, histogram, k):
        self.suppression_count = 0
        failing_classes = []
        for key,count in histogram.items():
            if count != 0:
                if count < k:
                    failing_classes.append(key)
                    self.suppression_count += count
        return len(failing_classes) == 0

    def merge_histograms(self, histograms):
        global_hist = {}
        for h in histograms:
            for key, count in h.items():
                global_hist[key] = global_hist.get(key, 0) + count
        return global_hist

    def get_final_binwidths(self, histogram, k):
        pass_nodes = []
        histogram_const = histogram.copy()

        total_nodes = sum(len(level) for level in self.tree)
        pbar = tqdm(total=total_nodes, desc="Marking Nodes", unit="node")

        while any(v is None for v in self.node_status.values()):
            histogram = histogram_const.copy()
            unmarked_levels = [
                i for i in range(len(self.tree))
                if any(self.node_status.get(tuple(node)) is None for node in self.tree[i])
            ]
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
                else:
                    if self.suppression_count < (self.suppression_limit * self.total_records) :
                        self._mark_subtree_pass(node,pbar)
                        pass_nodes.append(node)
                    else:
                        self._mark_parents_fail(node,pbar)
        
        pbar.close()
        self.find_best_rf(histogram_const, pass_nodes,k)

        if self.smallest_passing_rf is None:
            print("Warning: No passing RF found, falling back to initial bin widths")
            return list(self.tree[0][0])

        return self.smallest_passing_rf

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
                    if all(parent[i] <= current[i] for i in range(len(current))):
                        parent_key = tuple(parent)
                        if self.node_status.get(parent_key) is None:
                            self.node_status[parent_key] = 'fail'
                            q.append(parent)
                            if pbar: pbar.update(1)


    def find_best_rf(self, histogram, pass_nodes,k):
        best_node = None
        lowest_dm_star = float('inf')
        print("\nPassing nodes and their DM* values:")

        for node in pass_nodes:
            merged_hist = self.merge_equivalence_classes(histogram.copy(), list(node))

            low_count_sum = 0
            dm_star = 0

            for count in merged_hist.values():
                if count < k:
                    low_count_sum += count
                else:
                    dm_star += count * count

            dm_star += low_count_sum * low_count_sum

            print(f"Node: {list(node)}, DM*: {dm_star}")

            if dm_star < lowest_dm_star:
                lowest_dm_star = dm_star
                best_node = node

        self.smallest_passing_rf = best_node

        if best_node is not None:
            print(f"\nBest Node: {list(best_node)}, Final DM*: {lowest_dm_star}")
        else:
            print("No best node found.")

    def combine_generalized_chunks_to_csv(self, generalized_chunks, output_path='generalized_data.csv'):
        combined = pd.concat(generalized_chunks, ignore_index=True)
        combined.to_csv(output_path, index=False)
        print(f"Generalized data saved to {output_path}")
        return combined

    def get_suppressed_percent(self,node,histogram,k):
        histogram = self.merge_equivalence_classes(histogram, list(node))
        self.suppression_count = 0
        for _,count in histogram.items():
            if count != 0:
                if count < k:
                    self.suppression_count += count
        percent_suppressed = self.suppression_count / self.total_records
        percent_suppressed = percent_suppressed * 100
        return percent_suppressed