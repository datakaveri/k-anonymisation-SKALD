import math
from typing import List, Dict
from SKALD.categorical import CategoricalGeneralizer
import logging
logger = logging.getLogger("SKALD")


class OLA_1:
    """
    OLA_1 determines the minimal generalization (Ri) such that the
    estimated number of equivalence classes stays within memory limits.
    """

    # Allowed categorical hierarchy definitions
    _CATEGORICAL_LEVELS = {
        "blood group": 3,
        "gender": 2,
        "profession": 4,
    }

    def __init__(
        self,
        quasi_identifiers: List,
        n: int,
        max_equivalence_classes: int,
        multiplication_factors: Dict[str, int],
    ):
        # -------------------------
        # Validate constructor args
        # -------------------------
        if not quasi_identifiers:
            raise ValueError("quasi_identifiers cannot be empty")

        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")

        if not isinstance(max_equivalence_classes, int) or max_equivalence_classes <= 0:
            raise ValueError("max_equivalence_classes must be a positive integer")

        if not isinstance(multiplication_factors, dict):
            raise TypeError("multiplication_factors must be a dictionary")

        self.quasi_identifiers = quasi_identifiers
        self.n = n
        self.max_equivalence_classes = max_equivalence_classes
        self.multiplication_factors = multiplication_factors

        self.tree: List[List[List[int]]] = []
        self.node_status: Dict[tuple, str | None] = {}
        self.smallest_passing_ri = None

        self.categorical_generalizer = CategoricalGeneralizer()

    # ------------------------------------------------------------------
    # Equivalence class estimation
    # ------------------------------------------------------------------
    def calculate_equivalence_classes(self, bin_widths: List[int]) -> int:
        if len(bin_widths) != len(self.quasi_identifiers):
            raise ValueError("bin_widths length must match quasi_identifiers")

        num_classes = 1.0

        for qi, bw in zip(self.quasi_identifiers, bin_widths):
            if bw <= 0:
                raise ValueError(f"Invalid bin width {bw} for column '{qi.column_name}'")

            if qi.is_categorical:
                col = qi.column_name.lower()
                if col not in self._CATEGORICAL_LEVELS:
                    raise ValueError(
                        f"Unsupported categorical column '{qi.column_name}' in OLA_1"
                    )
                max_level = self._CATEGORICAL_LEVELS[col]
                if bw > max_level:
                    raise ValueError(
                        f"Categorical bin width {bw} exceeds max level {max_level} for '{qi.column_name}'"
                    )
                num_classes *= max_level // bw

            else:
                rng = qi.get_range()
                if rng <= 0:
                    raise ValueError(
                        f"Invalid numeric range for '{qi.column_name}'"
                    )
                num_classes *= math.ceil(rng / bw)

        return math.ceil(num_classes)
    def get_base_column_name(self,col_name: str) -> str:
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
    def build_tree(self):
        base = [1] * len(self.quasi_identifiers)
        self.tree = [[base]]
        self.node_status = {tuple(base): None}

        level = 0

        while True:
            next_level = []
            for node in self.tree[level]:
                for i, qi in enumerate(self.quasi_identifiers):
                    new_node = node.copy()

                    if qi.is_categorical:
                        col = qi.column_name.lower()
                        max_level = self._CATEGORICAL_LEVELS.get(col)
                        if max_level is None:
                            raise ValueError(
                                f"Unsupported categorical column '{qi.column_name}'"
                            )

                        if new_node[i] < max_level:
                            new_node[i] += 1

                    else:
                        col = self.get_base_column_name(qi.column_name)

                        if col not in self.multiplication_factors:
                            raise KeyError(
                                f"Missing multiplication factor for '{col}'"
                            )

                        factor = self.multiplication_factors[col]
                        max_val = qi.get_range()
                        new_node[i] = min(new_node[i] * factor, max_val)

                    t = tuple(new_node)
                    if t not in self.node_status:
                        next_level.append(new_node)
                        self.node_status[t] = None

            if not next_level:
                break

            self.tree.append(next_level)
            level += 1
        
        return self.tree

    # ------------------------------------------------------------------
    # Tree traversal
    # ------------------------------------------------------------------
    def find_smallest_passing_ri(self):
        while any(v is None for v in self.node_status.values()):
            for level in self.tree:
                for node in level:
                    key = tuple(node)
                    if self.node_status[key] is not None:
                        continue

                    classes = self.calculate_equivalence_classes(node)
                    if classes <= self.max_equivalence_classes:
                        self._mark_subtree_pass(node)
                    else:
                        self._mark_parents_fail(node)

        self._select_smallest_passing()
        return self.smallest_passing_ri

    def _mark_subtree_pass(self, node):
        self.node_status[tuple(node)] = "pass"
        if self.smallest_passing_ri is None or node < self.smallest_passing_ri:
            self.smallest_passing_ri = node

    def _mark_parents_fail(self, node):
        self.node_status[tuple(node)] = "fail"
        for level in self.tree:
            for parent in level:
                if all(p <= n for p, n in zip(parent, node)):
                    self.node_status.setdefault(tuple(parent), "fail")

    def _select_smallest_passing(self):
        passing = [
            node for level in self.tree
            for node in level
            if self.node_status.get(tuple(node)) == "pass"
        ]
        if not passing:
            raise ValueError(
                "No generalization satisfies equivalence class constraint"
            )
        self.smallest_passing_ri = [int(x) for x in min(passing)]

    # ------------------------------------------------------------------
    # Result accessors
    # ------------------------------------------------------------------
    def get_optimal_ri(self) -> List[int]:
        if self.smallest_passing_ri is None:
            raise ValueError("OLA_1 has not found a valid Ri")
        return self.smallest_passing_ri
