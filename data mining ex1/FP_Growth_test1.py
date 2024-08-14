import numpy as np
from collections import defaultdict

def loadDataSet():
    dataSet = []
    for line in open('./data mining ex1/T1014D100K.dat'):
        transaction = list(map(int, line.strip().split()))
        dataSet.append(transaction)
    return dataSet


class Node:
    def __init__(self, name, count, parent):
        self.name = name
        self.count = count
        self.parent = parent
        self.children = {}
        self.next = None

    def increment(self, count):
        self.count += count

    def display(self, depth=0):
        print('  ' * depth, self.name, ' ', self.count)
        for child in self.children.values():
            child.display(depth + 1)

def create_tree(data, min_support):
    header_table = defaultdict(int)
    for transaction in data:
        for item in transaction:
            header_table[item] += 1
    header_table = {k: v for k, v in header_table.items() if v >= min_support}
    # print(header_table)
    frequent_items = set(header_table.keys())
    if not frequent_items:
        return None, None
    for item in header_table:
        header_table[item] = [header_table[item], None]
    root = Node('Root', 1, None)
    for transaction, count in zip(data, [1]*len(data)):
        transaction = list(filter(lambda x: x in frequent_items, transaction))
        if len(transaction) > 0:
            current_node = root
            for item in sorted(transaction, key=lambda x: header_table[x][0], reverse=True):
                if item in current_node.children:
                    child_node = current_node.children[item]
                    child_node.increment(count)
                else:
                    child_node = Node(item, count, current_node)
                    current_node.children[item] = child_node
                    if header_table[item][1] is None:
                        header_table[item][1] = child_node
                    else:
                        update_header(header_table[item][1], child_node)
                current_node = child_node
    return root, header_table

def update_header(node_to_test, target_node):
    while node_to_test.next is not None:
        node_to_test = node_to_test.next
    node_to_test.next = target_node

def ascend_tree(node):
    path = []
    while node.parent is not None:
        path.append(node.name)
        node = node.parent
    return path

def find_prefix_path(base_path, tree_node):
    conditional_patterns = {}
    while tree_node is not None:
        path = ascend_tree(tree_node)
        if len(path) > 0:
            conditional_patterns[frozenset(path)] = tree_node.count
        tree_node = tree_node.next
    return conditional_patterns

def mine_tree(tree, header_table, min_support, prefix=set()):
    # header_table: item: [support, node]
    for item in sorted(header_table.keys(), key=lambda x: header_table[x][0]):
        print(item)
        new_prefix = prefix.copy()
        new_prefix.add(item)
        support = header_table[item][0]
        if support >= min_support:
            yield (new_prefix, support)
            conditional_tree_data = []
            node = header_table[item][1]
            while node is not None:
                path = ascend_tree(node)
                if len(path) > 0:
                    conditional_tree_data.append(path)
                node = node.next
            conditional_tree, conditional_header = create_tree(conditional_tree_data, min_support)
            if conditional_header is not None:
                for pattern in mine_tree(conditional_tree, conditional_header, min_support, new_prefix):
                    yield pattern

data = loadDataSet()

tree, header_table = create_tree(data, 600)
for pattern, support in mine_tree(tree, header_table, 600):
    print(pattern, support)
