import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)

np.set_printoptions(linewidth=1000)
np.set_printoptions(suppress=True)

base_file_name = './train/base_info.csv'
label_file_name = './train/entprise_info.csv'

# 读取文件
base = pd.read_csv(base_file_name)
label = pd.read_csv(label_file_name)
# print(base.head(1))

# 处理缺失值
missingDF = base.isnull().sum().sort_values(ascending=False).reset_index()
missingDF.columns = ['feature', 'miss_num']
# print(missingDF)
missingDF['miss_percentage'] = missingDF['miss_num'] / base.shape[0]
# 输出每列的缺失数量和缺失率
# print(missingDF)
# print(base.shape)
# 设置去除缺失率的阈值
thr = (1 - 0.2) * base.shape[0]
# 去除缺失率较大的特征
base = base.dropna(thresh=thr, axis=1)
# print(base.columns)

# 删除企业的经营地址、经营范围、经营场所等无用数据
del base['dom'], base['oploc'], base['opscope']
# print(base.columns)

# 合并base文件和标签文件
base1 = pd.merge(base, label, on='id', how='left')

# 删除id列，日期列，大写字母列， orgid:机构标识, jobid:职位标识
del base1['id'], base1['industryphy'], base1['opfrom'], base1['orgid'], base1['jobid']
# print(base1)
train_set = base1[base1['label'].notnull()]
y_train_label = train_set['label']
del train_set['label']
# print()
# print(train_set)

# 训练集的标签
train_label_list = list(train_set.columns)
# print(train_label_list)
has_nan = list(train_set.isnull().sum()>0)

# 含有少量缺失值的特征
nan_list = []
for i in range(len(has_nan)):
    if has_nan[i]:
        nan_list.append(train_label_list[i])
print(nan_list)
for i in train_label_list:
    print(train_set[i].describe())
for i in nan_list:
    # print(train_set[i].median())
    train_set[i].fillna(train_set[i].median(), inplace=True)
print(train_set.isnull().sum()>0)
test_set = base1[base1['label'].isnull()]
del test_set['label']
# print()
# print(test_set)
test_set_array = test_set.to_numpy()

# # 生成决策树
# Dt = tree.DecisionTreeClassifier()
# Dt = Dt.fit(train_set, y_train_label)
# y_test_label = Dt.predict(test_set)
# print(y_test_label)
