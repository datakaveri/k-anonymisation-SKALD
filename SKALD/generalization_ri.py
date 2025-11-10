import math
from SKALD.categorical import CategoricalGeneralizer

class OLA_1:
    """
    Implements the OLA (Optimal Lattice Anonymization) algorithm to determine the minimal 
    generalization (bin widths or hierarchy levels) for quasi-identifiers such that the number 
    of equivalence classes remains below a defined threshold(due to memory constraint).
    """

    def __init__(self, quasi_identifiers, n, max_equivalence_classes, multiplication_factors):
        """
        Initializes the OLA_1 object.

        Args:
            quasi_identifiers (List[QuasiIdentifier]): List of QI objects with domain info.
            n (int): Total number of records in the dataset.
            max_equivalence_classes (int): Threshold for maximum allowed equivalence classes.
            multiplication_factors (Dict[str, int]): Factors used to increase bin widths for numerical QIs.
        """
        self.quasi_identifiers = quasi_identifiers
        self.n = n
        self.max_equivalence_classes = max_equivalence_classes
        self.multiplication_factors = multiplication_factors
        self.tree = []  # Search tree of generalization levels
        self.smallest_passing_ri = None  # Holds smallest node that satisfies the constraints
        self.node_status = {}  # Stores pass/fail status for each node
        self.categorical_generalizer = CategoricalGeneralizer()

    def calculate_equivalence_classes(self, bin_widths):
        """
        Estimates the number of equivalence classes for a given generalization (bin_widths).

        Args:
            bin_widths (List[int]): Generalization levels or bin widths for each QI.

        Returns:
            int: Estimated number of equivalence classes.
        """
        num_classes = 1.0
        for qi, bin_width in zip(self.quasi_identifiers, bin_widths):
            if qi.is_categorical:
                # Hardcoded hierarchy levels for known categorical QIs
                if qi.column_name == 'Blood Group':
                    num_classes *= [8, 4, 1][int(bin_width)-1]
                elif qi.column_name == 'Profession':
                    num_classes *= [16, 4, 2, 1][int(bin_width)-1]
                elif qi.column_name.lower() == 'gender':
                    num_classes *= [2, 1][int(bin_width)-1]
            else:
                # For numeric QIs, use range/bin_width (unless bin_width is 0)
                num_classes *= (float(qi.get_range()) if bin_width == 0 else float(qi.get_range()) / float(bin_width))
        return math.ceil(num_classes)

    def build_tree(self):
        """
        Builds a lattice/tree of generalization combinations (bin widths per QI),
        starting from the minimal generalization [1, 1, ..., 1] up to the max levels.

        Returns:
            List[List[List[int]]]: Levels of the generalization tree.
        """
        base = [1] * len(self.quasi_identifiers)
        self.tree = [[base]]
        self.node_status = {tuple(base): None}

        level = 1
        while True:
            next_level = []
            for node in self.tree[level-1]:
                for i in range(len(node)):
                    new_node = node.copy()
                    qi = self.quasi_identifiers[i]

                    if qi.is_categorical:
                        # Max hierarchy levels for categorical QIs
                        if qi.column_name == "Blood Group":
                            max_level = 3
                        elif qi.column_name.lower() == "gender":
                            max_level = 2
                        else:
                            max_level = 4

                        if new_node[i] < max_level:
                            new_node[i] += 1
                            if tuple(new_node) not in self.node_status:
                                next_level.append(new_node)
                                self.node_status[tuple(new_node)] = None
                    else:
                        # For numeric QIs, multiply by factor until max range
                        max_val = qi.get_range()
                        if new_node[i] < max_val:
                            factor = self.multiplication_factors[qi.column_name[:-8]] if qi.is_encoded else self.multiplication_factors[qi.column_name]
                            new_node[i] = min(new_node[i] * factor, max_val)
                            if tuple(new_node) not in self.node_status:
                                next_level.append(new_node)
                                self.node_status[tuple(new_node)] = None

            if not next_level:
                break

            self.tree.append(next_level)
            level += 1

        return self.tree

    def get_precision(self, node):
        """
        Computes the generalization precision score for a node.
        Lower values indicate more specific (less generalized) configurations.

        Args:
            node (List[int]): Bin widths for each QI.

        Returns:
            float: Precision score.
        """
        precision = 0.0
        for i, bin_width in enumerate(node):
            qi = self.quasi_identifiers[i]

            if qi.is_categorical:
                if qi.column_name == "Blood Group":
                    max_level = 3
                elif qi.column_name.lower() == "gender":
                    max_level = 2
                else:
                    max_level = 4

                precision += bin_width / max_level
            else:
                base = self.multiplication_factors[qi.column_name[:-8]] if qi.is_encoded else self.multiplication_factors[qi.column_name]
                max_level = math.ceil(math.log(qi.get_range(), base)) + 2
                level = math.ceil(math.log(bin_width, base)) + 2
                precision += level / max_level

        return precision

    def find_smallest_passing_ri(self, n):
        """
        Traverses the generalization tree to find the smallest generalization (node)
        that satisfies the constraint on max equivalence classes.

        Args:
            n (int): Total number of records.

        Returns:
            List[int] or None: Smallest passing generalization node.
        """
        while any(status is None for status in self.node_status.values()):
            # Find unmarked levels
            unmarked_levels = [
                level for level in range(len(self.tree))
                if any(self.node_status.get(tuple(node)) is None for node in self.tree[level])
            ]
            if not unmarked_levels:
                break

            mid_level = unmarked_levels[len(unmarked_levels) // 2]
            sorted_nodes = sorted(
                [node for node in self.tree[mid_level] if self.node_status.get(tuple(node)) is None]
            )
            #print(sorted_nodes)
            for node in sorted_nodes:
                if self.node_status.get(tuple(node)) is not None:
                    continue
                #Print(node)
                classes = self.calculate_equivalence_classes(node)
                #print(f"No. of Equivalence classes for node : {node}, is {classes}" )
                #print(self.max_equivalence_classes)
                if classes <= self.max_equivalence_classes:
                    self._mark_subtree_pass(node)
                else:
                    self._mark_parents_fail(node)

        self._find_smallest_passing_node()
        return self.smallest_passing_ri

    def _mark_subtree_pass(self, node):
        """
        Marks the current node and all its children as 'pass'.

        Args:
            node (List[int]): Current passing node.
        """
        self.node_status[tuple(node)] = 'pass'
        if not self.smallest_passing_ri or node < self.smallest_passing_ri:
            self.smallest_passing_ri = node

        current_level_index = self._find_node_level(node)
        for next_level_index in range(current_level_index, len(self.tree)):
            for child_node in self.tree[next_level_index]:
                if all(child_node[i] > node[i] for i in range(len(node))):
                    if self.node_status.get(tuple(child_node)) is None:
                        self.node_status[tuple(child_node)] = 'pass'

    def _mark_parents_fail(self, node):
        """
        Marks the current node and all its parent nodes as 'fail'.

        Args:
            node (List[int]): Current failing node.
        """
        self.node_status[tuple(node)] = 'fail'

        current_level_index = -1
        for level_index, level_nodes in enumerate(self.tree):
            if node in level_nodes:
                current_level_index = level_index
                break

        while current_level_index > 0:
            current_level_index -= 1
            for parent_node in self.tree[current_level_index]:
                if all(parent_node[i] <= node[i] for i in range(len(node))):
                    if self.node_status.get(tuple(parent_node)) is None:
                        self.node_status[tuple(parent_node)] = 'fail'

    def _find_node_level(self, node):
        """
        Helper to locate the tree level of a node.

        Args:
            node (List[int]): The node to locate.

        Returns:
            int: Level index or -1 if not found.
        """
        for i, level in enumerate(self.tree):
            if node in level:
                return i
        return -1

    def _find_smallest_passing_node(self):
        """
        Finds the smallest generalization node (with minimal values) among all passing nodes.
        """
        for level in reversed(range(len(self.tree))):
            sorted_nodes = sorted(
                [node for node in self.tree[level] if self.node_status.get(tuple(node)) == 'pass']
            )
            if sorted_nodes:
                self.smallest_passing_ri = sorted_nodes[0]
                break

    def get_optimal_ri(self):
        """
        Returns the passing node with the best (lowest) precision score.

        Returns:
            List[int]: Best passing generalization node.

        Raises:
            ValueError: If no passing node is found.
        """
        best_precision = float("inf")
        optimal_ri = None
        for level in self.tree:
            for node in level:
                if self.node_status.get(tuple(node)) == "pass":
                    precision = self.get_precision(node)
                    if precision < best_precision:
                        best_precision = precision
                        optimal_ri = node
        print("Best Precision is", best_precision)

        if optimal_ri is None:
            raise ValueError("Failed to find a suitable initial bin width (Ri).")

        return optimal_ri

    def get_optimal_ri_array(self):
        """
        Returns all nodes in the tree that passed the equivalence class constraint.

        Returns:
            List[List[int]]: All passing generalization nodes.
        """
        return [
            node for level in self.tree
            for node in level if self.node_status.get(tuple(node)) == "pass"
        ]

    def print_tree_status(self):
        """
        Prints the structure of the tree and status (pass/fail) of each node.
        """
        print("\n=== Tree Structure & Status ===")
        for level_index, level in enumerate(self.tree):
            print(f"Level {level_index}:")
            for node in level:
                status = self.node_status.get(tuple(node), "Unknown")
                print(f"  Node: {node} -> Status: {status}")
        print("=== End Tree ===\n")
