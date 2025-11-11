import math
import json
import os
import pandas as pd
import itertools
import numpy as np
from tqdm import tqdm

from collections import defaultdict
from SKALD.categorical import CategoricalGeneralizer
from SKALD.encoder import get_encoding_dir


class OLA_2:
    """
    OLA_2 class implements an optimized lattice anonymization strategy for chunk-based k-anonymity.
    It supports both categorical and numerical quasi-identifiers, and builds a generalization tree to find
    the best trade-off between data utility and privacy within a suppression threshold.
    """

    def __init__(self, quasi_identifiers, total_records, suppression_limit, multiplication_factors,sensitive_parameter):
        self.quasi_identifiers = quasi_identifiers
        self.sensitive_parameter = sensitive_parameter
        self.sensitive_sets = None  
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
        #for i, qi in enumerate(self.quasi_identifiers):
        #    print(f"QI {i}: {qi.column_name}, is_categorical: {qi.is_categorical}")

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
                        elif qi.column_name.lower() == "gender":
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
            # Automatically use encoded column if needed
            if qi.is_encoded and not col.endswith("_encoded"):
                col += "_encoded"

            if qi.is_categorical:
                unique_values = dataset[col].dropna().unique()
                self.domains.append(sorted(unique_values.tolist()))
            else:
                self.domains.append(sorted(dataset[col].dropna().unique().tolist()))

    def get_index_tuple(self, row):
        indices = []
        for i, qi in enumerate(self.quasi_identifiers):
            col = qi.column_name
            if qi.is_encoded and not col.endswith("_encoded"):
                col += "_encoded"

            val = row[col]

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

        # Build histogram shape: number of bins per QI
        shape = []
        num_bin_info = []  # keep (col_min,col_max,n_bins) for numerics so we can clamp
        for qi, dom, bw in zip(self.quasi_identifiers, self.domains, bin_widths):
            if qi.is_categorical:
                shape.append(len(dom))
                num_bin_info.append(None)
            else:
                col_min, col_max = min(dom), max(dom)
                n_bins = int(np.ceil(max(1, (col_max - col_min + 1)) / max(1, int(bw))))
                shape.append(n_bins)
                num_bin_info.append((col_min, col_max, n_bins))

        histogram = np.zeros(shape, dtype=int)

        # Init sensitive sets array aligned with histogram
        self.sensitive_sets = np.empty(shape, dtype=object)
        it = np.nditer(histogram, flags=['multi_index', 'refs_ok'], op_flags=['writeonly'])
        for _ in it:
            self.sensitive_sets[it.multi_index] = set()

        def get_bin_index(row):
            indices = []
            for i, (qi, bw) in enumerate(zip(self.quasi_identifiers, bin_widths)):
                col = qi.column_name
                if qi.is_encoded and not col.endswith("_encoded"):
                    col += "_encoded"

                val = row[col]
                if qi.is_categorical:
                    indices.append(self.domains[i].index(val))
                else:
                    col_min, col_max, n_bins = num_bin_info[i]
                    bw = max(1, int(bw))
                    idx = int(np.floor((val - col_min) / bw))
                    # clamp
                    if idx < 0:
                        idx = 0
                    if idx >= n_bins:
                        idx = n_bins - 1
                    indices.append(idx)
            return tuple(indices)

        for _, row in chunk.iterrows():
            try:
                idx = get_bin_index(row)
                histogram[idx] += 1
                self.sensitive_sets[idx].add(row[self.sensitive_parameter])
            except Exception as e:
                # keep going for robustness
                # print(f"Skipping row due to error: {e}")
                continue

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
    def merge_sets_axis(self, array, axis, group_size):
        shape = array.shape
        new_size = (shape[axis] + group_size - 1) // group_size
        new_shape = list(shape)
        new_shape[axis] = new_size

        merged = np.empty(new_shape, dtype=object)
        for i in np.ndindex(*new_shape):
            merged[i] = set()

        for idx in np.ndindex(*shape):
            new_idx = list(idx)
            new_idx[axis] = idx[axis] // group_size
            merged[tuple(new_idx)] |= array[idx]

        return merged

    
    
    '''
    def print_histogram_classes(self, histogram, bin_widths):
        """
        Pretty-print all non-empty equivalence classes in the histogram.
        Shows count, number of distinct sensitive values, and the sensitive sets.
        """
        col_names = [qi.column_name for qi in self.quasi_identifiers]
        col_mins = []
        adj_bin_widths = []

        for qi, dom, bw in zip(self.quasi_identifiers, self.domains, bin_widths):
            if qi.is_categorical:
                col_mins.append(None)
                adj_bin_widths.append(None)
            else:
                col_mins.append(min(dom))
                adj_bin_widths.append(bw)

        print("\n=== Equivalence Classes ===")
        it = np.nditer(histogram, flags=['multi_index'])
        for x in it:
            if x > 0:  # non-empty bin
                idx = it.multi_index
                class_repr = []
                for i, (qi, dom, bw, min_val) in enumerate(zip(self.quasi_identifiers, self.domains, adj_bin_widths, col_mins)):
                    if qi.is_categorical:
                        val = dom[idx[i]]
                        class_repr.append(f"{col_names[i]}={val}")
                    else:
                        start = min_val + idx[i] * bw
                        end = start + bw
                        class_repr.append(f"{col_names[i]}=[{start}, {end})")

                # sensitive info
                sens_vals = self.sensitive_sets[idx]
                #print(f"{class_repr} -> count={int(x)}, sensitive_count={len(sens_vals)}")

    '''

    def merge_equivalence_classes(self, histogram, sensitive_sets, new_bin_widths):
        merged_hist = histogram.copy()
        merged_sets = sensitive_sets.copy()

        for i, (qi, level) in enumerate(zip(self.quasi_identifiers, new_bin_widths)):
            if not qi.is_categorical:
                group_size = max(1, int(level))
            else:
                # if domains are not built, fallback to axis length
                axis_len = merged_hist.shape[i]
                max_level = self._get_max_categorical_level(qi)
                # spread categories across levels; avoid div-by-zero
                denom = max(1, (max_level - int(level) + 1))
                group_size = max(1, axis_len // denom)

            merged_hist = self.merge_axis(merged_hist, i, group_size)
            merged_sets = self.merge_sets_axis(merged_sets, i, group_size)

        self.sensitive_sets = merged_sets
        return merged_hist, merged_sets




    def _get_max_categorical_level(self, qi):
        if qi.column_name == "Blood Group":
            return 3
        elif qi.column_name.lower() == "gender":
            return 2
        else:
            return 4  # default

    def check_k_anonymity(self, histogram, k, l):
        self.suppression_count = 0
        failing_mask = (histogram < k) & (histogram > 0)
        self.suppression_count = histogram[failing_mask].sum()
        if failing_mask.any():
            return False

        # l-diversity check
        if self.sensitive_sets is None:
            raise ValueError("Sensitive sets not initialized.")

        # Ensure shapes align; if not, rebuild a same-shaped array of empty sets
        if tuple(self.sensitive_sets.shape) != tuple(histogram.shape):
            filled = np.empty_like(histogram, dtype=object)
            it = np.nditer(histogram, flags=['multi_index'])
            for _ in it:
                filled[it.multi_index] = set()
            self.sensitive_sets = filled

        # len(None) safety
        diverse_mask = np.vectorize(lambda s: (len(s) if isinstance(s, set) else 0) >= l)(self.sensitive_sets)
        suppressed = (~diverse_mask) & (histogram > 0)

        self.suppression_count += histogram[suppressed].sum()
        return not suppressed.any()



    def merge_histograms(self, histograms):
        return np.sum(np.array(histograms), axis=0)

    def get_final_binwidths(self, histogram, k,l):
        pass_nodes = []
        histogram_const = histogram.copy()
        sensitive_sets_const = self.sensitive_sets.copy()
        total_nodes = sum(len(level) for level in self.tree)
        pbar = tqdm(total=total_nodes, desc="Marking Nodes", unit="node")
        #print(f"Histogram {histogram_const}")
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

                histogram, self.sensitive_sets = self.merge_equivalence_classes(histogram, sensitive_sets_const, node)


                if self.node_status[tuple(node)] is not None:
                    continue

                if self.check_k_anonymity(histogram, k,l):
                    self._mark_subtree_pass(node, pbar)
                    pass_nodes.append(node)
                elif self.suppression_count < (self.suppression_limit * self.total_records / 100):
                    self._mark_subtree_pass(node, pbar)
                    pass_nodes.append(node)
                else:
                    self._mark_parents_fail(node, pbar)

        pbar.close()
        self.find_best_rf(histogram_const, pass_nodes, k,l,sensitive_sets_const)
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

    def find_best_rf(self, histogram, pass_nodes, k, l, sensitive_sets):
        best_node = None
        lowest_dm_star = float('inf')
        best_num_eq_classes = None
        best_supp_count = None

        print("\nPassing nodes and their DM* values (with k-anonymity + l-diversity):")
        for node in pass_nodes:
            merged_hist, merged_sensitive = self.merge_equivalence_classes(
                histogram.copy(), sensitive_sets, list(node)
            )

            low_count_sum = np.sum(merged_hist[merged_hist < k])
            dm_star = np.sum(merged_hist[merged_hist >= k] ** 2) + low_count_sum ** 2
            num_eq_classes = np.sum(merged_hist > k)
            #print(merged_hist)

            self.sensitive_sets = merged_sensitive
            supp_count = self.suppression_count

            #print(f"Node: {list(node)}, DM*: {dm_star}, EQ Classes: {num_eq_classes},Suppressed Records: {supp_count}")
            
            eq_class_indices = np.argwhere(merged_hist > 0)
            #print("  Equivalence Classes (index → count):")
            for idx in eq_class_indices:
                count = merged_hist[tuple(idx)]
                #print(f"    {tuple(idx)} -> {count}")
            if dm_star <= lowest_dm_star:
                lowest_dm_star = dm_star
                best_node = node
                best_num_eq_classes = num_eq_classes
                best_supp_count = supp_count

        self.smallest_passing_rf = best_node
        self.lowest_dm_star = lowest_dm_star
        self.best_num_eq_classes = best_num_eq_classes
        self.best_suppression_count = best_supp_count


        if best_node is not None:
            print(f"\nBest Node: {list(best_node)}, Final DM*: {lowest_dm_star}")
        else:
            print("No best node found.")

        
    def get_equivalence_class_stats(self, histogram, bin_widths, k):
        """
        Returns a dictionary {eq_class_size: count}, only for retained (non-suppressed) equivalence classes.
        Suppressed classes (with size < k) are ignored.
        """
        merged_histogram, _ = self.merge_equivalence_classes(histogram, self.sensitive_sets, bin_widths)
        stats = defaultdict(int)

        flat = merged_histogram.flatten()
        for count in flat:
            if count >= k:
                stats[int(count)] += 1

        return dict(stats)


    def generalize_chunk(self, chunk, bin_widths):
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
                col_temp = col.removesuffix("_encoded")
                col_data = gen_chunk[col]
                encoding_dir = get_encoding_dir()
                encoding_file = os.path.join(
                    encoding_dir,
                    f"{col_temp.replace(' ', '_').lower()}_encoding.json"
                )

                if os.path.exists(encoding_file):
                    with open(encoding_file, "r") as f:
                        raw_map = json.load(f)
                    encoding_map = raw_map["encoding_map"]
                    decoding_map = {int(v): int(k) for k, v in encoding_map.items()} if isinstance(next(iter(encoding_map.values())), int) else {v: int(k) for k, v in encoding_map.items()}
                    multiplier = raw_map.get("multiplier", 1)

                    min_val = int(col_data.min())
                    max_val = int(col_data.max())

                    # ensure at least two edges
                    step = int(max(1, bw))
                    if min_val == max_val:
                        bin_edges = np.array([min_val, min_val + step])
                    else:
                        edges = list(range(min_val, max_val, step))
                        if not edges or edges[-1] < max_val + 1:
                            edges.append(max_val + 1)
                        bin_edges = np.array(edges, dtype=int)

                    labels = []
                    for i in range(len(bin_edges) - 1):
                        start_enc = int(bin_edges[i])
                        end_enc = int(bin_edges[i + 1] - 1)
                        if i == len(bin_edges) - 2:
                            end_enc = int(max_val)

                        if multiplier != 1:
                            start_dec = decoding_map.get(start_enc, start_enc) / multiplier
                            end_dec = decoding_map.get(end_enc, end_enc) / multiplier
                        else:
                            start_dec = decoding_map.get(start_enc, start_enc)
                            end_dec = decoding_map.get(end_enc, end_enc)

                        labels.append(f"[{start_dec}-{end_dec}]")

                    gen_chunk[col_temp] = pd.cut(col_data, bins=bin_edges, labels=labels, include_lowest=True, right=False)

                else:
                    min_val = int(col_data.min())
                    max_val = int(col_data.max())
                    step = int(max(1, bw))
                    if min_val == max_val:
                        bin_edges = np.array([min_val, min_val + step])
                    else:
                        start_edge = (min_val // step) * step
                        end_edge = ((max_val + step - 1) // step) * step + 1
                        bin_edges = np.arange(start_edge, end_edge, step, dtype=int)
                    labels = [f"[{int(bin_edges[i])}-{int(bin_edges[i + 1] - 1)}]" for i in range(len(bin_edges) - 1)]
                    gen_chunk[col] = pd.cut(col_data, bins=bin_edges, labels=labels, include_lowest=True, right=False)

        return gen_chunk

    
    def combine_generalized_chunks_to_csv(self, generalized_chunks, output_path='generalized_chunk1.csv'):
        combined = pd.concat(generalized_chunks, ignore_index=True)
        combined.to_csv(output_path, index=False)
        print(f"Generalized data saved to {output_path}")
        return combined

    def get_suppressed_percent(self, node, histogram, k):
        histogram,_ = self.merge_equivalence_classes(histogram, self.sensitive_sets,list(node))
        self.suppression_count = np.sum((histogram > 0) & (histogram < k))
        return (self.suppression_count / self.total_records) * 100
