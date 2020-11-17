import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import catboost as cb
from catboost import cv, Pool
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
import warnings

warnings.filterwarnings("ignore")
np.set_printoptions(linewidth=1000)
np.set_printoptions(suppress=True)

code_start = datetime.datetime.now()

base_file_name = './train/base_info.csv'        # base信息
annual_report_file_name = './train/annual_report_info.csv'  # 年报信息
label_file_name = './train/entprise_info.csv'   # 训练集的label信息
change_file_name = './train/change_info.csv'    # 变更信息
news_file_name = './train/news_info.csv'        # 新闻舆情信息
tax_file_name = './train/tax_info.csv'      # 纳税信息
other_file_name = './train/other_info.csv'  # 其它信息
test_file_name = './entprise_evaluate.csv'

# 读取文件
base = pd.read_csv(base_file_name)
report = pd.read_csv(annual_report_file_name)
label = pd.read_csv(label_file_name)

# 删除年报缺失值
thr = (1-0.3)*report.shape[0]
report_drop = report.dropna(thresh=thr, axis=1)
# 年报的表头
report_cols = list(report_drop.columns)
# 将类型为object的列填充
report_drop['BUSSTNAME'] = report_drop['BUSSTNAME'].fillna('无')
# 将object进行编码
le = preprocessing.LabelEncoder()
encode = le.fit(report_drop['BUSSTNAME'].unique())
report_drop['BUSSTNAME'] = le.transform(report_drop['BUSSTNAME'])

# 对其他列的缺失数据进行填充
for i in report_cols:
    if report_drop.isnull().sum()[i] > 0:
        report_drop[i].fillna(report_drop[i].median(), inplace=True)

# 将id之外的列的类型转换成int
report_drop[report_cols[1:]] = report_drop[report_cols[1:]].astype(int)
report_drop_group = (report_drop.groupby(by='id', sort=False).mean().astype(int))

# 将之前DataFrame的index作为现在的index
report_drop_group.reset_index(inplace=True)
# print(report_drop_group['id'])

# # 处理缺失值
# missingDF = base.isnull().sum().sort_values(ascending=False).reset_index()
# missingDF.columns = ['feature', 'miss_num']
# # print(missingDF)
# missingDF['miss_percentage'] = missingDF['miss_num'] / base.shape[0]

# 输出每列的缺失数量和缺失率
# print(missingDF)


# 设置去除缺失率的阈值
thr = (1 - 0.7) * base.shape[0]

# 去除缺失率较大的特征
base = base.dropna(thresh=thr, axis=1)
# print(base.columns)

# 删除企业的经营地址、经营范围、经营场所等无用数据、经营方式、经营期限止
del base['dom'], base['oploc'], base['opscope'], base['opform'], base['opto']

# 合并base和report文件
base_report = pd.merge(base, report_drop_group, on='id', how='left')
# print(base_report.columns)

# 合并base文件和标签文件
base1 = pd.merge(base_report, label, on='id', how='left')
# print(base1.columns)

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

# 训练集的表头
train_label_list = list(train_set.columns)

# 训练集填充缺失值，中位数填充
for i in train_label_list:
    if train_set.isnull().sum()[i] > 0:
        train_set[i].fillna(train_set[i].median(), inplace=True)
# label为1的数据集
label_one_set = train_set[train_set['label'] == 1]

# 标签为1的样本重复采集的次数
# times = 0
# for i in range(times):
#     train_set = pd.concat([train_set, label_one_set])
y_train_label = train_set['label']
del train_set['label']

# 分割出测试集
test_set = base1[base1['label'].isnull()]
del test_set['label']
test_columns_names = list(test_set.columns)

# 测试集填充缺失值,中位数填充
for i in test_columns_names:
    if test_set.isnull().sum()[i] > 0:
        test_set[i].fillna(test_set[i].median(), inplace=True)

categorical_features_indices = np.where(train_set.dtypes != np.float)[0]
# categorical_features_indices = np.array(range(0, train_set.shape[1]))
# print(categorical_features_indices)

# 生成训练集和验证集
x_train, x_validation, y_train, y_validation = train_test_split(train_set, y_train_label, train_size=0.95,
                                                                random_state=999)

# cross validation
params = {
    'iterations': 1000,
    'learning_rate': 0.1,
    'eval_metric': 'Accuracy',
    'random_seed': 42,
    'max_depth': 5,
    'logging_level': 'Silent',
    'use_best_model': True
}

train_pool = Pool(x_train, y_train, cat_features=categorical_features_indices)
validate_pool = Pool(x_validation, y_validation, cat_features=categorical_features_indices)

# train_pool = Pool(x_train, y_train)
# validate_pool = Pool(x_validation, y_validation)

# # cv交叉验证
# cv_params = {
#     'loss_function': 'Logloss',
#     'iterations': 500,
#     'custom_loss': 'F1',
#     'random_seed': 99,
#     'learning_rate': 0.1,
# }
# cv_data = cv(
#     params=cv_params,
#     pool=train_pool,
#     fold_count=10,
#     type='Classical',
#     shuffle=True,
#     partition_random_seed=8,
#     verbose=True
#
# )

# # earlystop-model
# earlystop_params = params.copy()
# earlystop_params.update(
#     {
#         'od_type': 'Iter',
#         'od_wait': 100
#     }
# )
# earlystop_model = cb.CatBoostClassifier(**earlystop_params)
# earlystop_model.fit(train_pool, eval_set=validate_pool)
# print('earlystop model validation accuracy: {:.4}'.format(
#     accuracy_score(y_validation, earlystop_model.predict(x_validation))
# ))
# test_label = earlystop_model.predict(test_set)

# # hyperopt metric optimize
# space = {
#     'iterations': hp.randint('iterations', 1000),
#     'random_seed': hp.randint('random_seed', 100),
#     'eval_metric': hp.choice('eval_metric', ['Accuracy', 'F1', 'Recall']),
#     'learning_rate': hp.choice('learning_rate', [0.02, 0.05, 0.1]),
#     'depth': hp.randint('max_depth', 11)
# }
#
#
# def hyperopt_train_test(params):
#     clf = cb.CatBoostClassifier(**params)
#     recall = cross_val_score(clf, x_train, y_train, scoring='recall').mean()
#     f1 = cross_val_score(clf, x_train, y_train, scoring='f1').mean()
#     precision = cross_val_score(clf, x_train, y_train, scoring='precision').mean()
#     return recall+f1+precision
#
#
# def f(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
#
# trials = Trials()
# best = fmin(
#     f,
#     space,
#     algo=tpe.suggest,
#     max_evals=100,
#     trials=trials
# )
# print("best:  ", end='')
# print(best)
# for trial in trials.trials:
#     print(trial['result'])

# best_model
best_model = cb.CatBoostClassifier(**params)
best_model.fit(train_pool, eval_set=validate_pool)
print('Best model validation accuracy: {:.4}'.format(
    accuracy_score(y_validation, best_model.predict(x_validation))
))
test_label = best_model.predict(test_set)

print(list(test_label).count(0))

# 输出每个特征值的权重
importances = best_model.get_feature_importance(prettified=True)
print(importances)

# 输出验证集的F1分数
metrics = best_model.eval_metrics(data=train_pool, metrics=['Logloss', 'F1', 'Precision', 'Recall'])
print("Logloss: ", min(metrics['Logloss']))
print("f1 score: ", max(metrics['F1']))
print("Precision: ", max(metrics['Precision']))
print("Recall: ", max(metrics['Recall']))

code_end = datetime.datetime.now()
print("code run time: ", code_end - code_start)

# 将预测结果保存并写入文件
final_test_set['label'] = test_label

# 需要提交的顺序
test_submit = pd.read_csv(test_file_name)
del test_submit['score']

# 合并提交文件和预测结果
test_submit = pd.merge(test_submit, final_test_set, on='id', how='left')
test_submit.rename(columns={'label': 'score'}, inplace=True)

# 写入文件
save_test_res_name = './test_label_folder/base_report_f1_recall_precision_depth_5_1117.csv'
test_submit.to_csv(save_test_res_name, sep=',', header=True, index=False)
