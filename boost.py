import pandas as pd
import numpy as np
from decisiontree import DTree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import catboost as cb

np.set_printoptions(linewidth=1000)
np.set_printoptions(suppress=True)

base_file_name = './train/base_info.csv'
annual_report_file_name = './train/annual_report_info.csv'
label_file_name = './train/entprise_info.csv'
test_file_name = './entprise_evaluate.csv'

# 读取文件
base = pd.read_csv(base_file_name)
annual_report = pd.read_csv(annual_report_file_name)
label = pd.read_csv(label_file_name)

# 处理缺失值
missingDF = base.isnull().sum().sort_values(ascending=False).reset_index()
missingDF.columns = ['feature', 'miss_num']
# print(missingDF)
missingDF['miss_percentage'] = missingDF['miss_num'] / base.shape[0]
# 输出每列的缺失数量和缺失率
# print(missingDF)
# print(base.shape)

# 设置去除缺失率的阈值
thr = (1 - 0.4) * base.shape[0]

# 去除缺失率较大的特征
base = base.dropna(thresh=thr, axis=1)
# print(base.columns)

# 删除企业的经营地址、经营范围、经营场所等无用数据
del base['dom'], base['oploc'], base['opscope']

# 合并base文件和标签文件
base1 = pd.merge(base, label, on='id', how='left')

# 删除id列，日期列，大写字母列， orgid:机构标识, jobid:职位标识
del base1['industryphy'], base1['opfrom'], base1['orgid'], base1['jobid']
# print(base1.columns)

# 用于存储预测标签
temp_test_set = base1[base1['label'].isnull()]
final_test_set = pd.DataFrame(columns=['id', 'label'])
final_test_set['id'] = temp_test_set['id']
final_test_set['label'] = temp_test_set['label']
del base1['id']

# 分割出训练集
train_set = base1[base1['label'].notnull()]
y_train_label = train_set['label']

# 训练集的标签
train_label_list = list(train_set.columns)
# print(train_label_list)
has_nan = list(train_set.isnull().sum() > 0)

# 含有少量缺失值的特征
nan_list = []
for i in range(len(has_nan)):
    if has_nan[i]:
        nan_list.append(train_label_list[i])

# 填充缺失值
for i in nan_list:
    # print(train_set[i].median())
    train_set[i].fillna(train_set[i].mean(), inplace=True)
# print(train_set.isnull().sum()>0)
del train_set['label']
# print(train_set.isnull().sum())
# print(y_train_label)
# print(train_label_list)

test_set = base1[base1['label'].isnull()]
del test_set['label']
test_columns_names = list(test_set.columns)
test_nan_list = list(test_set.isnull().sum() > 0)

# 需要填充的列
columns_to_fill = []
for i in range(len(test_columns_names)):
    if test_nan_list[i]:
        columns_to_fill.append(test_columns_names[i])
# 测试集填充缺失值
for i in columns_to_fill:
    test_set[i].fillna(test_set[i].mean(), inplace=True)

# print(test_set)
test_set_array = test_set.to_numpy()

# 生成二叉决策树,训练集中类别数量
# num_0:   13884
# num_1:   981
# dt = DecisionTreeClassifier(max_depth=6)

dt = cb.CatBoostClassifier()
dt.fit(train_set, y_train_label)
test_label = dt.predict(test_set)

# # 利用训练集，对模型进行评估
# train_label_predict = dt.predict(train_set)
# tp, fp, fn, tn = 0, 0, 0, 0
# y_train_label_list = list(y_train_label)
# for i in range(len(y_train_label_list)):
#     if train_label_predict[i]==0:
#         if y_train_label_list[i]==0:
#             tp += 1
#         else:
#             fp += 1
#     else:
#         if y_train_label_list[i]==0:
#             fn += 1
#         else:
#             tn += 1
# print("tp:   ", tp)
# print("fp:   ", fp)
# print("fn:   ", fn)
# print("tn:   ", tn)
# print("tp+tn:  ", tp+tn)
# print("0类预测正确率：  ", tp/y_train_label_list.count(0))
# print("1类预测正确率：  ", tn/y_train_label_list.count(1))
# print("总体预测正确率： ", (tn+tp)/len(y_train_label_list))

# 将预测结果保存并写入文件
print(list(test_label).count(0))
final_test_set['label'] = test_label

# 需要提交的顺序
test_submit = pd.read_csv(test_file_name)
del test_submit['score']

# 合并提交文件和预测结果
test_submit = pd.merge(test_submit, final_test_set, on='id', how='left')
test_submit.rename(columns={'label': 'score'}, inplace=True)

# 写入文件
save_test_res_name = './test_label_folder/entprise_evaluate1_catboost_classifier_1102_depth_drop0.4.csv'
test_submit.to_csv(save_test_res_name, sep=',', header=True, index=False)
