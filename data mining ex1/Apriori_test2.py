import numpy as np
import itertools
def loadDataSet():
    dataSet = []
    for line in open('./data mining ex1/T1014D1K.dat'):
        transaction = list(map(int, line.strip().split()))
        dataSet.append(transaction)
    return dataSet
from collections import defaultdict, Counter

def generate_candidates(frequent_items, k):
    # 生成候选项集
    candidates = set()
    for a in frequent_items:
        for b in frequent_items:
            union_list = a.union(b)
            if len(union_list) == k and a != b and all(frozenset(subset) in frequent_items.keys() for subset in itertools.combinations(union_list, k-1)):
            # if len(union_list) == k and a != b:
                candidates.add(union_list)
    # 使用哈希表存储候选项集，加速项集是否是频繁项的判断
    hash_table = defaultdict(int)
    for transaction in candidates:
        for item in transaction:
            hash_table[item] += 1
            
    frequent_candidates = {transaction: hash_table[transaction] for transaction in candidates}
    return frequent_candidates

def apriori_hash(data, min_support):
    # 计算所有单项的支持度
    item_counts = Counter(item for transaction in data for item in transaction)
    frequent_items = {frozenset([item]): count for item, count in item_counts.items() if count >= min_support}
    # 如果没有频繁项，直接返回
    if not frequent_items:
        return {}
    print(len(frequent_items))
    # 逐层增加项数并计算支持度
    k = 1
    while True:
        k += 1
        candidate_items = generate_candidates(frequent_items, k)
        if not candidate_items:
            break
            
        # 遍历数据集并更新候选项集的支持度计数
        item_counts = defaultdict(int)
        for transaction in data:
            updated_items = [item for item in candidate_items if item.issubset(transaction)]
            for item in updated_items:
                item_counts[item] += 1
        # 保留支持度大于等于min_support的项集
        frequent_items = {item: count for item, count in item_counts.items() if count >= min_support}
        print(len(frequent_items))
        if not frequent_items:
            break
    return frequent_items

data = loadDataSet()
min_support = 6
frequent_items = apriori_hash(data, min_support)
print(frequent_items)
