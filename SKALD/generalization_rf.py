# SKALD/generalization_rf.py

import math
import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List
from tqdm import tqdm
import glob
import logging
logger = logging.getLogger("SKALD")

from SKALD.categorical import CategoricalGeneralizer
from SKALD.encoder import get_encoding_dir


class OLA_2:
    """
    OLA_2 implements RF selection for chunk-based k-anonymity
    with optional l-diversity and suppression control.

    IMPORTANT:
    - This class is STATEFUL
    - Public API MUST NOT CHANGE (core.py depends on it)
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def __init__(
        self,
        quasi_identifiers,
        total_records: int,
        suppression_limit: float,
        multiplication_factors: dict,
        sensitive_parameter: str | None,
        enable_l_diversity: bool = True,
    ):
        if not quasi_identifiers:
            raise ValueError("quasi_identifiers cannot be empty")

        if total_records <= 0:
            raise ValueError("total_records must be > 0")

        if not (0 <= suppression_limit <= 100):
            raise ValueError("suppression_limit must be in [0,100]")

        if enable_l_diversity and not sensitive_parameter:
            raise ValueError("sensitive_parameter required for l-diversity")

        self.quasi_identifiers = quasi_identifiers
        self.total_records = total_records
        self.suppression_limit = suppression_limit
        self.multiplication_factors = multiplication_factors
        self.sensitive_parameter = sensitive_parameter
        self.enable_l_diversity = enable_l_diversity

        self.tree = []
        self.node_status = {}

        self.domains = []
        self.sensitive_sets = None

        self.smallest_passing_rf = None
        self.lowest_dm_star = float("inf")
        self.best_num_eq_classes = None
        self.best_suppression_count = None
        self.suppression_count = 0

        self.categorical_generalizer = CategoricalGeneralizer()

    def get_base_column_name(self, col_name: str) -> str:
        if col_name.endswith("_scaled_encoded"):
            return col_name[:-15]
        elif col_name.endswith("_encoded"):
            return col_name[:-8]
        elif col_name.endswith("_scaled"):
            return col_name[:-7]
        else:
            return col_name
    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------
    def build_tree(self, initial_ri: List[int]):
        if not initial_ri:
            raise ValueError("initial_ri cannot be empty")

        self.tree = [[initial_ri]]
        self.node_status = {tuple(initial_ri): None}

        while True:
            next_level = []
            for node in self.tree[-1]:
                for i, qi in enumerate(self.quasi_identifiers):
                    new_node = node.copy()

                    if qi.is_categorical:
                        max_level = self._get_max_categorical_level(qi)
                        if new_node[i] < max_level:
                            new_node[i] += 1
                    else:
                        base_col = self.get_base_column_name(qi.column_name)
                        if base_col not in self.multiplication_factors:
                            raise KeyError(
                                f"Missing multiplication factor for '{base_col}'"
                            )
                        factor = self.multiplication_factors[base_col]
                        new_node[i] = min(new_node[i] * factor, qi.get_range())

                    t = tuple(new_node)
                    if t not in self.node_status:
                        next_level.append(new_node)
                        self.node_status[t] = None

            if not next_level:
                break
            self.tree.append(next_level)

        return self.tree

    # ------------------------------------------------------------------
    # Histogram building (REQUIRED by pipeline)
    # ------------------------------------------------------------------
    def build_domains(self, dataset: pd.DataFrame):
        self.domains = []

        for qi in self.quasi_identifiers:
            col = qi.column_name  # already effective column

            if col not in dataset.columns:
                raise KeyError(f"Column '{col}' missing while building domains")

            values = dataset[col].dropna().unique().tolist()

            if not values:
                raise ValueError(f"No valid values for column '{col}'")

            self.domains.append(sorted(values))


    def process_chunk(self, chunk: pd.DataFrame, bin_widths: List[int]) -> np.ndarray:
        if not isinstance(chunk, pd.DataFrame) or chunk.empty:
            raise ValueError("chunk must be a non-empty DataFrame")

        if not self.domains:
            self.build_domains(chunk)

        shape = []
        num_bin_info = []

        for qi, dom, bw in zip(self.quasi_identifiers, self.domains, bin_widths):
            if qi.is_categorical:
                shape.append(len(dom))
                num_bin_info.append(None)
            else:
                col_min, col_max = min(dom), max(dom)
                n_bins = int(
                    math.ceil(
                        max(1, (col_max - col_min + 1)) / max(1, int(bw))
                    )
                )
                shape.append(n_bins)
                num_bin_info.append((col_min, col_max, n_bins))

        histogram = np.zeros(shape, dtype=int)

        self.sensitive_sets = np.empty(shape, dtype=object)
        for idx in np.ndindex(*shape):
            self.sensitive_sets[idx] = set()

        for _, row in chunk.iterrows():
            try:
                idx = self._get_bin_index(row, bin_widths, num_bin_info)
                histogram[idx] += 1
                if self.enable_l_diversity:
                    self.sensitive_sets[idx].add(row[self.sensitive_parameter])
            except Exception:
                continue

        return histogram


    def _get_bin_index(self, row, bin_widths, num_bin_info):
        indices = []

        for i, (qi, bw) in enumerate(zip(self.quasi_identifiers, bin_widths)):
            col = qi.column_name  # already effective column

            val = row[col]

            if qi.is_categorical:
                indices.append(self.domains[i].index(val))
            else:
                col_min, _, n_bins = num_bin_info[i]
                idx = int((val - col_min) // max(1, int(bw)))
                indices.append(max(0, min(idx, n_bins - 1)))

        return tuple(indices)


    # ------------------------------------------------------------------
    # Histogram merging
    # ------------------------------------------------------------------
    def merge_histograms(self, histograms: List[np.ndarray]) -> np.ndarray:
        if not histograms:
            raise ValueError("No histograms to merge")
        return np.sum(np.array(histograms), axis=0)

    # ------------------------------------------------------------------
    # RF evaluation
    # ------------------------------------------------------------------

    def get_final_binwidths(self, histogram, k: int, l: int):
        if histogram is None:
            raise ValueError("histogram cannot be None")

        pass_nodes = []
        base_hist = histogram.copy()
        base_sets = self.sensitive_sets.copy()

        total_nodes = sum(len(level) for level in self.tree)
        pbar = tqdm(total=total_nodes, desc="Evaluating nodes", unit="node")

        while any(v is None for v in self.node_status.values()):
            histogram = base_hist.copy()
            unmarked_levels = [i for i in range(len(self.tree)) if any(self.node_status.get(tuple(node)) is None for node in self.tree[i])]
            if not unmarked_levels:
                break
            
            mid_level = unmarked_levels[len(unmarked_levels) // 2]
            sorted_nodes = sorted(
                [node for node in self.tree[mid_level] if self.node_status[tuple(node)] is None], reverse=True
            )

            if sorted_nodes:
                node = sorted_nodes[len(sorted_nodes) // 2]
                key = tuple(node)
                if self.node_status[key] is not None:
                    print("Node %s already marked, skipping.", node)
                    pbar.update(1)
                    continue

                merged_hist, merged_sets = self.merge_equivalence_classes(
                    base_hist.copy(), base_sets.copy(), node
                )

                passes = self.check_k_anonymity(merged_hist, k, l)
                if passes or self.suppression_count <= (
                    self.suppression_limit * self.total_records
                ):
                    logger.info("suppression_count for %s: %d", node, self.suppression_count)
                    self.node_status[key] = "pass"
                    pass_nodes.append(node)
                    self._mark_subtree_pass(node, pbar)

                else:
                    self.node_status[key] = "fail"
                    self._mark_parents_fail(node, pbar)

                pbar.update(1)

        pbar.close()

        if not pass_nodes:
            raise ValueError("No node satisfies k-anonymity constraints")

        self.find_best_rf(base_hist, pass_nodes, k, l, base_sets)
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
                    parent_key = tuple(parent)
                    if all(parent[i] <= current[i] for i in range(len(current))) and self.node_status.get(parent_key) is None:
                        self.node_status[parent_key] = 'fail'
                        q.append(parent)
                        if pbar: pbar.update(1)
    # ------------------------------------------------------------------
    # k / l checks
    # ------------------------------------------------------------------
    def check_k_anonymity(self, merged_histogram, k: int, l: int):

        self.suppression_count = int(
            np.sum(merged_histogram[(merged_histogram > 0) & (merged_histogram < k)])
        )

        if self.suppression_count > 0:
            return False
        '''
        if not self.enable_l_diversity:
            return True

        for idx in np.ndindex(histogram.shape):
            if histogram[idx] > 0 and len(self.sensitive_sets[idx]) < l:
                self.suppression_count += histogram[idx]
                return False
        '''
        return True

    # ------------------------------------------------------------------
    # Metrics & stats (REQUIRED by core.py)
    # ------------------------------------------------------------------
    def get_equivalence_class_stats(self, histogram, bin_widths, k):
        if histogram is None or self.sensitive_sets is None:
            raise RuntimeError("Histogram or sensitive_sets not initialized")

        merged_hist, _ = self.merge_equivalence_classes(
            histogram, self.sensitive_sets, bin_widths
        )

        stats = defaultdict(int)
        for count in merged_hist.flatten():
            if count >= k:
                stats[int(count)] += 1

        return dict(stats)

    def get_suppressed_percent(self, node, histogram, k):
        merged_hist, _ = self.merge_equivalence_classes(
            histogram, self.sensitive_sets, list(node)
        )
        suppressed = np.sum((merged_hist > 0) & (merged_hist < k))
        return (suppressed / self.total_records) * 100

    # ------------------------------------------------------------------
    # Histogram merging helpers
    # ------------------------------------------------------------------
    def merge_equivalence_classes(self, histogram, sensitive_sets, node):
        merged_hist = histogram
        merged_sets = sensitive_sets

        for axis, (qi, level) in enumerate(zip(self.quasi_identifiers, node)):
            group = max(1, int(level))
            merged_hist = self._merge_axis(merged_hist, axis, group)
            merged_sets = self._merge_sets_axis(merged_sets, axis, group)

        self.sensitive_sets = merged_sets
        return merged_hist, merged_sets

    @staticmethod
    def _merge_axis(arr, axis, group):
        pad = (-arr.shape[axis]) % group
        if pad:
            pad_shape = list(arr.shape)
            pad_shape[axis] = pad
            arr = np.concatenate((arr, np.zeros(pad_shape, dtype=int)), axis=axis)

        new_shape = arr.shape[:axis] + (-1, group) + arr.shape[axis + 1 :]
        return np.sum(arr.reshape(new_shape), axis=axis + 1)

    @staticmethod
    def _merge_sets_axis(arr, axis, group):
        new_shape = list(arr.shape)
        new_shape[axis] = math.ceil(arr.shape[axis] / group)
        out = np.empty(new_shape, dtype=object)
        for idx in np.ndindex(*new_shape):
            out[idx] = set()

        for idx in np.ndindex(arr.shape):
            new_idx = list(idx)
            new_idx[axis] //= group
            out[tuple(new_idx)] |= arr[idx]

        return out

    # ------------------------------------------------------------------
    # RF selection
    # ------------------------------------------------------------------
    def find_best_rf(self, histogram, pass_nodes, k, l, sensitive_sets):
        for node in pass_nodes:
            merged_hist, _ = self.merge_equivalence_classes(
                histogram.copy(), sensitive_sets.copy(), node
            )

            dm_star = np.sum(merged_hist[merged_hist >= k] ** 2)

            if dm_star < self.lowest_dm_star:
                self.lowest_dm_star = dm_star
                self.smallest_passing_rf = node
                self.best_num_eq_classes = int(np.sum(merged_hist >= k))


    def generalize_chunk(self, chunk, bin_widths, s_list):

        gen_chunk = chunk.copy(deep=False)

        for idx, (qi, bw) in enumerate(zip(self.quasi_identifiers, bin_widths)):
            s = int(s_list[idx]) if idx < len(s_list) else 0
            col = qi.column_name

            # -------------------------
            # HANDLE CATEGORICAL QIs
            # -------------------------
            if qi.is_categorical:
                level = int(bw)
                col_key = qi.column_name.strip().lower()
                if col_key == "blood group":
                    mapper = self.categorical_generalizer.generalize_blood_group
                elif col_key == "gender":
                    mapper = self.categorical_generalizer.generalize_gender
                elif col_key == "profession":
                    mapper = self.categorical_generalizer.generalize_profession
                else:
                    mapper = lambda x, _: x
                gen_chunk[col] = gen_chunk[col].map(lambda x: mapper(x, level))
                continue

            # -------------------------
            # HANDLE NUMERICAL QIs
            # -------------------------
            original_col = col.removesuffix("_scaled_encoded") \
                if col.endswith("_scaled_encoded") else \
                col.removesuffix("_encoded")

            # ------------------------------------
            # DETERMINE ENCODED SERIES
            # ------------------------------------
            if qi.is_encoded:
                enc_series = gen_chunk[col].astype("int64")

                encoding_file = os.path.join(
                    get_encoding_dir(),
                    f"{original_col.replace(' ', '_').lower()}_encoding.json"
                )

                with open(encoding_file, "r") as f:
                    raw_map = json.load(f)

                encoding_map = raw_map["encoding_map"]
                decoding_map = {int(v): int(k) for k, v in encoding_map.items()}

            else:
                enc_series = gen_chunk[col].astype("int64")
                decoding_map = None

            # ------------------------------------
            # BUILD BIN EDGES IN ENCODED SPACE
            # ------------------------------------
            min_val = int(enc_series.min())
            max_val = int(enc_series.max())
            step = max(1, int(bw))

            encoded_edges = list(range(min_val, max_val + 1, step))
            encoded_edges.append(encoded_edges[-1] + step)
            encoded_edges = np.array(encoded_edges, dtype=int)

            # ------------------------------------
            # DECODE BIN EDGES (STILL SCALED DOMAIN)
            # ------------------------------------
            if decoding_map:
                decoded_edges = np.array(
                    [decoding_map.get(v, v) for v in encoded_edges],
                    dtype=float
                )
            else:
                decoded_edges = encoded_edges.astype(float)

            # ------------------------------------
            # INVERSE SCALING TO ORIGINAL DOMAIN
            # ------------------------------------
            if s > 0:
                decoded_edges = decoded_edges * (10 ** s)
            elif s < 0:
                decoded_edges = np.round(decoded_edges * (10 ** s), abs(s))
            # s == 0 → identity

            # ------------------------------------
            # BUILD LABELS
            # ------------------------------------
            labels = []
            for i in range(len(decoded_edges) - 1):
                left = decoded_edges[i]
                right = decoded_edges[i + 1]
                labels.append(f"[{left}-{right})")

            # ------------------------------------
            # APPLY GENERALISATION
            # ------------------------------------
            gen_chunk[original_col] = pd.cut(
                enc_series,
                bins=encoded_edges,
                labels=labels,
                include_lowest=True,
                right=False
            )

        return gen_chunk



    def get_suppressed_percent(self, node, histogram, k):
        histogram,_ = self.merge_equivalence_classes(histogram, self.sensitive_sets,list(node))
        self.suppression_count = np.sum((histogram > 0) & (histogram < k))
        return (self.suppression_count / self.total_records) * 100

    def _print_failing_equivalence_classes(self, histogram, k, l,node):
        print("\n========== Failing Equivalence Classes ==========\n")

        # 1. Print histogram bins < k  (k-anonymity failures)
        failing_k = np.argwhere((histogram > 0) & (histogram < k))
        new_bin_widths = list(node)
        if failing_k.size > 0:
            print(" K-ANONYMITY FAILURES (count < k):")
            for idx in failing_k:
                count = histogram[tuple(idx)]
                desc = self.describe_equivalence_class(tuple(idx), new_bin_widths)
                print(f" {desc}  → count = {count}")
        else:
            print("✔ No k-anonymity failures.")

        # If l-diversity is disabled, stop here
        if not self.enable_l_diversity:
            print("\nL-diversity disabled → skipping sensitive-set failures.\n")
            return

        # 2. L-diversity failures
        print("\n L-DIVERSITY FAILURES:")
        for idx in np.ndindex(histogram.shape):
            count = histogram[idx]
            if count == 0:
                continue
            sens_set = self.sensitive_sets[idx]
            if len(sens_set) < l:
                print(f"  Index {idx} -> count={count}, sensitive_values={sens_set}")

    def describe_equivalence_class(self, idx_tuple, bin_widths):
        """
        Converts a histogram index into human-readable bin intervals per QI.
        """
        desc = []

        for i, (qi, bin_w) in enumerate(zip(self.quasi_identifiers, bin_widths)):
            dom = self.domains[i]
            idx = idx_tuple[i]

            # ---------- CATEGORICAL ----------
            if qi.is_categorical:
                try:
                    value = dom[idx]
                    desc.append(f"{qi.column_name} = {value}")
                except Exception:
                    desc.append(f"{qi.column_name} = <invalid_index {idx}>")

            # ---------- NUMERIC ----------
            else:
                # numeric QI domain is the sorted list of possible values
                try:
                    col_min, col_max = min(dom), max(dom)
                except Exception:
                    desc.append(f"{qi.column_name} = <domain_error>")
                    continue

                bw = int(bin_w)
                start = col_min + idx * bw
                end = start + bw - 1

                # clamp numeric range
                if start < col_min: start = col_min
                if end > col_max: end = col_max

                desc.append(f"{qi.column_name} ∈ [{start}–{end}]")

        return ", ".join(desc)
    
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_max_categorical_level(qi):
        if qi.column_name == "Blood Group":
            return 3
        if qi.column_name.lower() == "gender":
            return 2
        return 4
