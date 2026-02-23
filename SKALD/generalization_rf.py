# SKALD/generalization_rf.py

import math
import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
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
        use_variance_il: bool = True,
        lambda1: Optional[float] = 0.5,
        lambda2: Optional[float] = 0.25,
        lambda3: Optional[float] = 0.25,
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
        self.use_variance_il = bool(use_variance_il)
        self.lambda1 = 0.5 if lambda1 is None else float(lambda1)
        self.lambda2 = 0.25 if lambda2 is None else float(lambda2)
        self.lambda3 = 0.25 if lambda3 is None else float(lambda3)

        self.tree = []
        self.node_status = {}

        self.domains = []
        self.sensitive_sets = None

        self.smallest_passing_rf = None
        self.lowest_dm_star = float("inf")
        self.best_num_eq_classes = None
        self.best_suppression_count = None
        self.suppression_count = 0
        self.top_rf_nodes = []
        self.original_qi_df = None
        self.domain_weights = []
        self.max_dm_star = 1.0
        self.max_precision_loss = 1.0
        self.max_variance_il = 1.0

        self.categorical_generalizer = CategoricalGeneralizer()

    def set_original_qi_df(self, df: pd.DataFrame):
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            self.original_qi_df = None
            return
        self.original_qi_df = df.copy()

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
    # Sparse histogram helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_sparse(histogram) -> bool:
        return isinstance(histogram, dict)

    @staticmethod
    def _merge_sparse_histograms(h1: Dict[Tuple[int, ...], int],
                                 h2: Dict[Tuple[int, ...], int]) -> Dict[Tuple[int, ...], int]:
        out = dict(h1)
        for k, v in h2.items():
            out[k] = out.get(k, 0) + v
        return out

    @staticmethod
    def _merge_sparse_equivalence_classes(
        histogram: Dict[Tuple[int, ...], int],
        sensitive_sets: Optional[Dict[Tuple[int, ...], set]],
        node: List[int],
    ) -> Tuple[Dict[Tuple[int, ...], int], Optional[Dict[Tuple[int, ...], set]]]:
        merged_hist: Dict[Tuple[int, ...], int] = {}
        merged_sets: Optional[Dict[Tuple[int, ...], set]] = {} if sensitive_sets is not None else None

        for idx, count in histogram.items():
            merged_idx = tuple(
                max(0, idx[i] // max(1, int(node[i]))) for i in range(len(idx))
            )
            merged_hist[merged_idx] = merged_hist.get(merged_idx, 0) + count
            if merged_sets is not None and sensitive_sets is not None:
                if idx in sensitive_sets:
                    merged_sets.setdefault(merged_idx, set()).update(sensitive_sets[idx])

        return merged_hist, merged_sets
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


    def process_chunk(self, chunk: pd.DataFrame, bin_widths: List[int]) -> dict:
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

        histogram: Dict[Tuple[int, ...], int] = {}
        self.sensitive_sets = {}

        for _, row in chunk.iterrows():
            try:
                idx = self._get_bin_index(row, bin_widths, num_bin_info)
                histogram[idx] = histogram.get(idx, 0) + 1
                if self.enable_l_diversity:
                    self.sensitive_sets.setdefault(idx, set()).add(row[self.sensitive_parameter])
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
    def merge_histograms(self, histograms):
        if not histograms:
            raise ValueError("No histograms to merge")

        first = histograms[0]
        if self._is_sparse(first):
            merged = {}
            for h in histograms:
                merged = self._merge_sparse_histograms(merged, h)
            return merged

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
        if self._is_sparse(merged_histogram):
            self.suppression_count = int(
                sum(v for v in merged_histogram.values() if 0 < v < k)
            )
        else:
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
        if self._is_sparse(merged_hist):
            for count in merged_hist.values():
                if count >= k:
                    stats[int(count)] += 1
        else:
            for count in merged_hist.flatten():
                if count >= k:
                    stats[int(count)] += 1

        return dict(stats)

    def get_suppressed_percent(self, node, histogram, k):
        merged_hist, _ = self.merge_equivalence_classes(
            histogram, self.sensitive_sets, list(node)
        )
        if self._is_sparse(merged_hist):
            suppressed = sum(1 for v in merged_hist.values() if 0 < v < k)
        else:
            suppressed = np.sum((merged_hist > 0) & (merged_hist < k))
        return (suppressed / self.total_records) * 100

    # ------------------------------------------------------------------
    # Histogram merging helpers
    # ------------------------------------------------------------------
    def merge_equivalence_classes(self, histogram, sensitive_sets, node):
        if self._is_sparse(histogram):
            merged_hist, merged_sets = self._merge_sparse_equivalence_classes(
                histogram, sensitive_sets, node
            )
            self.sensitive_sets = merged_sets
            return merged_hist, merged_sets

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
    @staticmethod
    def _get_suppression_count_for_histogram(hist, k: int) -> int:
        if isinstance(hist, dict):
            return int(sum(v for v in hist.values() if 0 < v < k))
        return int(np.sum(hist[(hist > 0) & (hist < k)]))

    def _get_top_node(self) -> List[int]:
        if not self.tree:
            return []
        dims = len(self.tree[0][0])
        return [max(node[i] for level in self.tree for node in level) for i in range(dims)]

    def _compute_domain_weights(self) -> List[float]:
        if self.domain_weights:
            return self.domain_weights

        importances = []
        for i, qi in enumerate(self.quasi_identifiers):
            if self.domains and i < len(self.domains):
                domain_size = max(1, len(self.domains[i]))
            else:
                if qi.is_categorical:
                    domain_size = max(1, int(self._get_max_categorical_level(qi)))
                else:
                    domain_size = max(1, int(qi.get_range()))
            importances.append(1.0 / math.log(domain_size + 1.0))

        total = sum(importances)
        if total <= 0:
            self.domain_weights = [1.0 / len(self.quasi_identifiers)] * len(self.quasi_identifiers)
        else:
            self.domain_weights = [v / total for v in importances]
        return self.domain_weights

    def compute_precision_loss(self, generalization_vector: List[int]) -> float:
        weights = self._compute_domain_weights()
        loss = 0.0
        for i, (qi, bw) in enumerate(zip(self.quasi_identifiers, generalization_vector)):
            if qi.is_categorical:
                max_level = max(1, int(self._get_max_categorical_level(qi)))
                denom = max(1, max_level - 1)
                level_loss = (max(1, int(bw)) - 1) / denom
            else:
                max_bw = max(1, int(qi.get_range()))
                denom = max(1, max_bw - 1)
                level_loss = (max(1, int(bw)) - 1) / denom
            loss += weights[i] * min(1.0, max(0.0, float(level_loss)))
        return float(loss)

    def _make_group_keys(self, generalization_vector: List[int], df: pd.DataFrame) -> pd.DataFrame:
        key_df = pd.DataFrame(index=df.index)
        for i, (qi, bw) in enumerate(zip(self.quasi_identifiers, generalization_vector)):
            col = qi.column_name
            if col not in df.columns:
                key_df[f"_g_{i}"] = -1
                continue

            if qi.is_categorical:
                dom = self.domains[i] if self.domains and i < len(self.domains) else sorted(df[col].dropna().unique().tolist())
                dom_map = {v: idx for idx, v in enumerate(dom)}
                cat_idx = df[col].map(dom_map).fillna(-1).astype(int)
                key_df[f"_g_{i}"] = (cat_idx // max(1, int(bw))).astype(int)
            else:
                series = pd.to_numeric(df[col], errors="coerce")
                if self.domains and i < len(self.domains) and self.domains[i]:
                    col_min = float(min(self.domains[i]))
                else:
                    col_min = float(series.min()) if series.notna().any() else 0.0
                num_idx = ((series - col_min) // max(1, int(bw))).fillna(-1).astype(int)
                key_df[f"_g_{i}"] = num_idx
        return key_df

    def compute_variance_il(self, generalization_vector: List[int], original_df: Optional[pd.DataFrame] = None) -> float:
        if original_df is None:
            original_df = self.original_qi_df
        if original_df is None or original_df.empty:
            return 0.0

        weights = self._compute_domain_weights()
        key_df = self._make_group_keys(generalization_vector, original_df)
        group_keys = [tuple(row) for row in key_df.to_numpy()]
        variance_il = 0.0

        for i, qi in enumerate(self.quasi_identifiers):
            if qi.is_categorical:
                continue
            col = qi.column_name
            if col not in original_df.columns:
                continue

            col_vals = pd.to_numeric(original_df[col], errors="coerce")
            temp = pd.DataFrame({
                "_group": group_keys,
                "_value": col_vals
            })
            temp = temp[temp["_value"].notna()]
            if temp.empty:
                continue

            grouped = temp.groupby("_group")["_value"]
            # SSE = variance(population) * count
            sse = ((grouped.var(ddof=0).fillna(0.0) * grouped.count()).sum())
            variance_il += weights[i] * float(sse)

        return float(variance_il)

    def _compute_dm_star_for_node(self, histogram, sensitive_sets, node, k: int) -> Tuple[float, int, int]:
        merged_hist, _ = self.merge_equivalence_classes(
            histogram.copy(), sensitive_sets.copy(), node
        )

        if self._is_sparse(merged_hist):
            dm_star = float(sum(v * v for v in merged_hist.values() if v >= k))
            num_eq = int(sum(1 for v in merged_hist.values() if v >= k))
        else:
            dm_star = float(np.sum(merged_hist[merged_hist >= k] ** 2))
            num_eq = int(np.sum(merged_hist >= k))

        suppression_count = self._get_suppression_count_for_histogram(merged_hist, k)
        return dm_star, num_eq, suppression_count

    @staticmethod
    def _safe_normalize(value: float, max_value: float) -> float:
        if max_value <= 0:
            return 0.0
        return min(1.0, max(0.0, float(value) / float(max_value)))

    def _prepare_metric_normalizers(self, histogram, sensitive_sets, k: int):
        top_node = self._get_top_node()
        if not top_node:
            self.max_dm_star = 1.0
            self.max_precision_loss = 1.0
            self.max_variance_il = 1.0
            return

        dm_star_top, _, _ = self._compute_dm_star_for_node(histogram, sensitive_sets, top_node, k)
        precision_top = self.compute_precision_loss(top_node)
        variance_top = self.compute_variance_il(top_node, self.original_qi_df) if self.use_variance_il else 0.0

        self.max_dm_star = max(1.0, float(dm_star_top))
        self.max_precision_loss = max(1e-9, float(precision_top))
        self.max_variance_il = max(1e-9, float(variance_top)) if self.use_variance_il else 1.0

    def compute_weighted_score(self, node: List[int], dm_star: float, precision_loss: float, variance_il: float) -> float:
        ndm = self._safe_normalize(dm_star, self.max_dm_star)
        nprec = self._safe_normalize(precision_loss, self.max_precision_loss)
        nvar = self._safe_normalize(variance_il, self.max_variance_il) if self.use_variance_il else 0.0

        if not self.use_variance_il:
            return ndm

        return (
            self.lambda1 * ndm
            + self.lambda2 * nprec
            + self.lambda3 * nvar
        )

    def find_best_rf(self, histogram, pass_nodes, k, l, sensitive_sets):
        scored_nodes = []
        self._compute_domain_weights()
        self._prepare_metric_normalizers(histogram, sensitive_sets, k)

        for node in pass_nodes:
            dm_star, num_eq, suppression_count = self._compute_dm_star_for_node(
                histogram, sensitive_sets, node, k
            )
            precision_loss = self.compute_precision_loss(node)
            variance_il = self.compute_variance_il(node, self.original_qi_df) if self.use_variance_il else 0.0
            weighted_score = self.compute_weighted_score(node, dm_star, precision_loss, variance_il)

            scored_nodes.append({
                "node": [int(x) for x in node],
                "dm_star": float(dm_star),
                "precision_loss": float(precision_loss),
                "variance_il": float(variance_il),
                "weighted_score": float(weighted_score),
                "num_equivalence_classes": int(num_eq),
                "suppression_count": int(suppression_count),
            })

        if self.use_variance_il:
            scored_nodes.sort(
                key=lambda x: (
                    x["weighted_score"],
                    x["dm_star"],
                    x["suppression_count"],
                    -x["num_equivalence_classes"],
                    x["node"],
                )
            )
        else:
            scored_nodes.sort(
                key=lambda x: (
                    x["dm_star"],
                    x["suppression_count"],
                    -x["num_equivalence_classes"],
                    x["node"],
                )
            )

        self.top_rf_nodes = scored_nodes[:5]

        if self.top_rf_nodes:
            best = self.top_rf_nodes[0]
            self.lowest_dm_star = best["dm_star"]
            self.smallest_passing_rf = best["node"]
            self.best_num_eq_classes = best["num_equivalence_classes"]
            self.best_suppression_count = best["suppression_count"]
            logger.info(
                "Best OLA_2 node=%s dm_star=%.4f precision_loss=%.6f variance_il=%.6f weighted_score=%.6f",
                best.get("node"),
                float(best.get("dm_star", 0.0)),
                float(best.get("precision_loss", 0.0)),
                float(best.get("variance_il", 0.0)),
                float(best.get("weighted_score", best.get("dm_star", 0.0))),
            )

    def get_top_rf_nodes(self, top_n: int = 5):
        if top_n <= 0:
            raise ValueError("top_n must be > 0")
        return self.top_rf_nodes[:top_n]


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
        histogram, _ = self.merge_equivalence_classes(histogram, self.sensitive_sets, list(node))
        if self._is_sparse(histogram):
            self.suppression_count = sum(1 for v in histogram.values() if 0 < v < k)
        else:
            self.suppression_count = np.sum((histogram > 0) & (histogram < k))
        return (self.suppression_count / self.total_records) * 100

    def _print_failing_equivalence_classes(self, histogram, k, l,node):
        print("\n========== Failing Equivalence Classes ==========\n")

        # 1. Print histogram bins < k  (k-anonymity failures)
        new_bin_widths = list(node)
        if self._is_sparse(histogram):
            failing_items = [(idx, v) for idx, v in histogram.items() if 0 < v < k]
            if failing_items:
                print(" K-ANONYMITY FAILURES (count < k):")
                for idx, count in failing_items:
                    desc = self.describe_equivalence_class(tuple(idx), new_bin_widths)
                    print(f" {desc}  → count = {count}")
            else:
                print("✔ No k-anonymity failures.")
        else:
            failing_k = np.argwhere((histogram > 0) & (histogram < k))
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
        if self._is_sparse(histogram):
            for idx, count in histogram.items():
                if count == 0:
                    continue
                sens_set = self.sensitive_sets.get(idx, set())
                if len(sens_set) < l:
                    print(f"  Index {idx} -> count={count}, sensitive_values={sens_set}")
        else:
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
