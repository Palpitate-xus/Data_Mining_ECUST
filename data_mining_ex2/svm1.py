from sklearn import svm
import numpy as np
from sklearn.metrics import precision_score, recall_score

def loadDataSet():
    dataSet = []
    for line in open('./data mining ex2/train_sample.txt'):
        transaction = list(map(float, line.strip().split()))
        dataSet.append(transaction)
    return dataSet

def loadDataLabel():
    dataLabel = [int(x.strip()) for x in open('./data mining ex2/train_sample_label.txt').readlines()]
    return dataLabel

def loadTestSet():
    dataSet = []
    for line in open('./data mining ex2/test_sample.txt'):
        transaction = list(map(float, line.strip().split()))
        dataSet.append(transaction)
    return dataSet

def loadTestLabel():
    dataLabel = [int(x.strip()) for x in open('./data mining ex2/test_sample_label.txt').readlines()]
    return dataLabel

# 创建一个训练数据集
X = np.array(loadDataSet())
y = np.array(loadDataLabel())

# 创建一个测试样本
test_sample = np.array(loadTestSet())
def test_svm(kernel='poly', C=1.0, degree=3, gamma='scale', coef0=0.0, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None):
    # 创建一个SVM分类器，并选择线性内核
    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, coef0=coef0, tol=tol, cache_size=cache_size, class_weight=class_weight, verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape, break_ties=break_ties, random_state=random_state)
    # tol（float，默认值为1e-3）：控制模型的停止容忍度（tolerance）。当优化算法的变化小于tol时，算法将停止迭代。
    # C（float，默认值为1.0）：正则化参数，控制错误项的惩罚强度。较小的C值会导致决策边界较简单，容忍更多的错误分类，可能导致欠拟合。较大的C值会导致决策边界更复杂，可能导致过拟合。
    # 线性核（Linear Kernel）/ 多项式核（Polynomial Kernel）/ 径向基函数（RBF）核（Radial Basis Function Kernel）/ Sigmoid核（Sigmoid Kernel）
    # 训练SVM分类器
    clf.fit(X, y)
    # 使用训练好的模型进行预测
    y_pred = clf.predict(test_sample)
    # 计算准确率和查全率
    precision = precision_score(loadTestLabel(), y_pred)
    recall = recall_score(loadTestLabel(), y_pred)
    print("=============\nkernel=%s, C=%f, gamma=%s, coef0=%f, degree=%d, tol=%lf, cache_size=%f, class_weight=%s, verbose=%s, max_iter=%d, decision_function_shape=%s, break_ties=%s, random_state=%s"%(kernel, C, gamma, coef0, degree, tol, cache_size, class_weight, verbose, max_iter, decision_function_shape, break_ties, random_state))
    print("准确率: %s, 查全率: %s"%(precision, recall))
    
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
gamma = ['scale', 'auto']
decision_function_shape = ['ovo', 'ovr']

# for i in kernel:
#     test_svm(kernel=i)

# for i in gamma:
#     test_svm(gamma=i)

# for i in decision_function_shape:
#     test_svm(decision_function_shape=i)
    
# for i in np.arange(0.1, 20, 0.5):
#     test_svm(C=i)
    
# for i in np.arange(0.001, 1, 0.01):
#     test_svm(tol=i)

# for i in range(1, 20):
#     test_svm(kernel='poly', degree=i) # 多项式阶数