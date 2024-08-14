import numpy as np

def loadDataSet(filename):
    dataSet = []
    for line in open('./data mining ex1/' + filename + '.dat'):
        transaction = list(map(int, line.strip().split()))
        dataSet.append(transaction)
    return dataSet

def apriori(data, min_support, file):
    file.write('min_support = ' + str(min_support) + '\n')
    # 将数据集中的每个项转换成集合
    data_sets = [set(row) for row in data]
    # 获取所有不重复的项
    unique_items = sorted(set.union(*data_sets))
    # 生成所有可能的一项集
    itemsets = [{item} for item in unique_items]
    # 计算支持度并保留满足最小支持度要求的项集
    frequent_itemsets = []
    for itemset in itemsets:
        support = sum(1 for row in data_sets if itemset.issubset(row)) / len(data_sets)
        if support >= min_support:
            frequent_itemsets.append((itemset, support))
    # print(len(frequent_itemsets))
    file.write('频繁1项集的个数： ' + str(len(frequent_itemsets)) + '\n')
    # for i in frequent_itemsets:
        # file.write(str(i) + '\n')
    # 组合不同长度的频繁项集，生成候选项集
    k = 2
    while frequent_itemsets:
        candidate_itemsets = set()
        for i in range(len(frequent_itemsets)):
            for j in range(i + 1, len(frequent_itemsets)):
                itemset1, _ = frequent_itemsets[i]
                itemset2, _ = frequent_itemsets[j]
                # 取并集
                union = itemset1.union(itemset2)
                if len(union) == k and union not in candidate_itemsets:
                    candidate_itemsets.add(frozenset(union))

        # 计算支持度并保留满足最小支持度要求的项集
        frequent_itemsets = []
        for itemset in candidate_itemsets:
            support = sum(1 for row in data_sets if itemset.issubset(row)) / len(data_sets)
            if support >= min_support:
                frequent_itemsets.append((itemset, support))
        
        file.write('频繁' + str(k) + '项集的个数： ' + str(len(frequent_itemsets)) + '\n')
        # for i in frequent_itemsets:
        #     file.write(str(i) + '\n')
        k += 1

    # 返回所有频繁项集
    return [itemset for itemset, _ in frequent_itemsets]

files = ['T1014D1K', 'T1014D10K', 'T1014D50K', 'T1014D100K']
supports = [0.006, 0.008, 0.010]
for i in files:
    filename = 'new' + i
    for min_support in supports:
        f = open(filename + str(min_support) + '.txt', 'w', encoding='UTF-8')
        apriori(loadDataSet(i), min_support, f)
        f.close()
