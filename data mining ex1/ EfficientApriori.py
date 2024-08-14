import numpy as np
from itertools import combinations

def generate_candidate_itemsets(itemset, length):
    return set([i.union(j) for i in itemset for j in itemset if len(i.union(j)) == length])

def prune_itemsets(itemset, transaction_list, min_support, freq_set):
    local_set = defaultdict(int)
    for item in itemset:
        for transaction in transaction_list:
            if item.issubset(transaction):
                freq_set[item] += 1
                local_set[item] += 1
    pruned_itemset = set()
    for item, count in local_set.items():
        support = float(count) / len(transaction_list)
        if support >= min_support:
            pruned_itemset.add(item)
    return pruned_itemset

def apriori(transaction_list, min_support, n_threads=4):
    freq_set = defaultdict(int)
    large_set = dict()
    items = set()
    for transaction in transaction_list:
        for item in transaction:
            items.add(frozenset([item]))
    k = 2
    pool = ThreadPool(n_threads)
    while True:
        candidates = generate_candidate_itemsets(items, k)
        if not candidates:
            break
        pruned_candidates = pool.map(partial(prune_itemsets, transaction_list=transaction_list, min_support=min_support, freq_set=freq_set), [candidates[i:i+n_threads] for i in range(0, len(candidates), n_threads)])
        items = set.union(*pruned_candidates)
        k += 1
    pool.close()
    for itemset in items:
        large_set[itemset] = freq_set[itemset]
    return large_set


min_support = 0.4
n_threads = 4

result = apriori(transaction_list, min_support, n_threads)

print(result)