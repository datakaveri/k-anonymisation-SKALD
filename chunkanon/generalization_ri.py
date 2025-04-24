import math
from chunkanon.categorical import CategoricalGeneralizer

class OLA_1:
    def __init__(self, quasi_identifiers, n, max_equivalence_classes, doubling_step=2):
        self.quasi_identifiers = quasi_identifiers
        self.n = n
        self.max_equivalence_classes = max_equivalence_classes
        self.doubling_step = doubling_step
        self.tree = []
        self.smallest_passing_ri = None
        self.node_status = {}
        self.categorical_generalizer = CategoricalGeneralizer()

    def calculate_equivalence_classes(self, n, bin_widths):
        num_classes = 1
        for qi, bin_width in zip(self.quasi_identifiers, bin_widths):
            if qi.is_categorical:
                if qi.column_name == 'Blood Group':
                    num_classes *= [8, 4, 1][int(bin_width)]
                elif qi.column_name == 'Profession':
                    num_classes *= [16, 4, 2, 1][int(bin_width)]
            else:
                if qi.column_name == 'BMI':
                    num_classes *= (qi.get_range() * 10 if bin_width == 0 else qi.get_range() / bin_width)
                else:
                    num_classes *= (qi.get_range() if bin_width == 0 else qi.get_range() / bin_width)

        return math.ceil(num_classes)

    def build_tree(self):
        base = [0] * len(self.quasi_identifiers)
        self.tree = [[base]]
        self.node_status = {tuple(base): None}

        level = 0
        while True:
            next_level = []
            for node in self.tree[level]:
                for i in range(len(node)):
                    new_node = node.copy()

                    if self.quasi_identifiers[i].is_categorical:
                        max_level = 2 if self.quasi_identifiers[i].column_name == "Blood Group" else 3
                        if new_node[i] < max_level:
                            new_node[i] += 1
                            if tuple(new_node) not in self.node_status:
                                next_level.append(new_node)
                                self.node_status[tuple(new_node)] = None
                    else:
                        max_val = self.quasi_identifiers[i].get_range()
                        if new_node[i] < max_val:
                            if new_node[i] == 0:
                                new_node[i] = 1
                            else:
                                factor = 10 if self.quasi_identifiers[i].column_name == "PIN Code" else \
                                         5 if self.quasi_identifiers[i].column_name == "encoded_PIN" else self.doubling_step
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
        precision = 0
        for i, bin_width in enumerate(node):
            qi = self.quasi_identifiers[i]
            if qi.is_categorical:
                max_levels = 3 if qi.column_name == 'Blood Group' else 4
                precision += (bin_width + 1) / max_levels
            else:
                if qi.column_name == 'PIN Code':
                    base = 10
                elif qi.column_name == 'encoded_PIN':
                    base = 5
                else:
                    base = self.doubling_step

                max_levels = math.ceil(math.log(qi.get_range(), base)) + 2
                level = 1 if bin_width == 0 else math.ceil(math.log(bin_width, base)) + 2
                precision += level / max_levels

        return precision

    def find_smallest_passing_ri(self, n):
        while any(status is None for status in self.node_status.values()):
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

            for node in sorted_nodes:
                if self.node_status.get(tuple(node)) is not None:
                    continue

                classes = self.calculate_equivalence_classes(n, node)
                if classes <= self.max_equivalence_classes:
                    self._mark_subtree_pass(node)
                else:
                    self._mark_parents_fail(node)

        self._find_smallest_passing_node()
        return self.smallest_passing_ri

    def _mark_subtree_pass(self, node):
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
        self.node_status[tuple(node)] = 'fail'
        current_level_index = self._find_node_level(node)
        for level_index in reversed(range(current_level_index)):
            for parent_node in self.tree[level_index]:
                if all(parent_node[i] < node[i] for i in range(len(node))):
                    if self.node_status.get(tuple(parent_node)) is None:
                        self.node_status[tuple(parent_node)] = 'fail'

    def _find_node_level(self, node):
        for i, level in enumerate(self.tree):
            if node in level:
                return i
        return -1

    def _find_smallest_passing_node(self):
        for level in reversed(range(len(self.tree))):
            sorted_nodes = sorted(
                [node for node in self.tree[level] if self.node_status.get(tuple(node)) == 'pass']
            )
            if sorted_nodes:
                self.smallest_passing_ri = sorted_nodes[0]
                break

    def get_optimal_ri(self):
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
        return optimal_ri

    def get_optimal_ri_array(self):
        return [
            node for level in self.tree
            for node in level if self.node_status.get(tuple(node)) == "pass"
        ]
