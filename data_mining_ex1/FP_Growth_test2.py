from collections import defaultdict

class Node:
    def __init__(self, item=None, count=0, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = defaultdict(Node)

class FPTree:
    def __init__(self, transactions, minsup):
        self.minsup = minsup
        self.root = Node()
        self.header_table = defaultdict(list)
        
        for transaction in transactions:
            self.add(transaction)

    def add(self, transaction):
        current_node = self.root
        for item in transaction:
            next_node = current_node.children[item]
            next_node.parent = current_node
            next_node.count += 1
            current_node = next_node

            if item not in self.header_table:
                self.header_table[item] = []
            self.header_table[item].append(next_node)

    def get_conditional_pattern_base(self, item):
        conditional_pattern_base = []
        nodes = self.header_table[item]
        for node in nodes:
            prefix_path = []
            current_node = node.parent
            while current_node.parent is not None:
                prefix_path.append(current_node.item)
                current_node = current_node.parent
            if prefix_path:
                conditional_pattern_base.append((prefix_path, node.count))

        return conditional_pattern_base

    def fp_growth(self, prefix, items):
        frequent_items = []
        for item, nodes in sorted(self.header_table.items(), key=lambda x: x[1][0].count):
            support = sum(node.count for node in nodes)
            if support < self.minsup:
                continue

            frequent_itemset = prefix + [item]
            frequent_items.append((frequent_itemset, support))

            conditional_pattern_base = self.get_conditional_pattern_base(item)
            conditional_tree = FPTree([transaction for transaction, _ in conditional_pattern_base], self.minsup)
            if not conditional_tree.root.children:
                continue

            conditional_frequent_items = conditional_tree.fp_growth(frequent_itemset, [node.item for node in nodes])
            frequent_items.extend(conditional_frequent_items)

        return frequent_items

def fp_growth(transactions, minsup):
    fptree = FPTree(transactions, minsup)
    return fptree.fp_growth([], [])

def loadDataSet():
    dataSet = []
    for line in open('./data mining ex1/T1014D1K.dat'):
        transaction = list(map(int, line.strip().split()))
        dataSet.append(transaction)
    return dataSet

# Example usage
transactions = loadDataSet()

minsup = 6

frequent_itemsets = fp_growth(transactions, minsup)
print(frequent_itemsets)
print('Number of frequent itemsets:', len(frequent_itemsets))
