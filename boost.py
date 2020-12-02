import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import catboost as cb
from catboost import cv, Pool
import lightgbm as lgb
from sklearn.preprocessing import normalize, scale
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_curve
from sklearn.model_selection import cross_val_score
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
import warnings

warnings.filterwarnings("ignore")
np.set_printoptions(linewidth=1000, suppress=True)

code_start = datetime.datetime.now()

base_file_name = './train/base_info.csv'  # base信息
annual_report_file_name = './train/annual_report_info.csv'  # 年报信息
label_file_name = './train/entprise_info.csv'  # 训练集的label信息
change_file_name = './train/change_info.csv'  # 变更信息
news_file_name = './train/news_info.csv'  # 新闻舆情信息
tax_file_name = './train/tax_info.csv'  # 纳税信息
other_file_name = './train/other_info.csv'  # 其它信息
entprise_evaluate_file_name = './entprise_evaluate.csv'

# 读取文件
base = pd.read_csv(base_file_name)
report = pd.read_csv(annual_report_file_name)
change = pd.read_csv(change_file_name)
news = pd.read_csv(news_file_name)
tax = pd.read_csv(tax_file_name)
other = pd.read_csv(other_file_name)
label = pd.read_csv(label_file_name)
entprise_evaluate = pd.read_csv(entprise_evaluate_file_name)


# 定义drop函数，通过nan的数量将columns删除
def drop_col_by_nan(df, ratio=0.05):
    cols = []
    for col in df.columns:
        if df[col].isna().mean() >= (1 - ratio):
            cols.append(col)
    return cols


# 删除文件中nan0.99以上的cols
base_info = base.drop(drop_col_by_nan(base, 0.01), axis=1)
base_info['district_FLAG1'] = (base_info['orgid'].fillna('').apply(lambda x: str(x)[:6]) ==
                               base_info['oplocdistrict'].fillna('').apply(lambda x: str(x)[:6])).astype(int)
base_info['district_FLAG2'] = (base_info['orgid'].fillna('').apply(lambda x: str(x)[:6]) ==
                               base_info['jobid'].fillna('').apply(lambda x: str(x)[:6])).astype(int)
base_info['district_FLAG3'] = (base_info['oplocdistrict'].fillna('').apply(lambda x: str(x)[:6]) ==
                               base_info['jobid'].fillna('').apply(lambda x: str(x)[:6])).astype(int)

base_info['person_SUM'] = base_info[['empnum', 'parnum', 'exenum']].sum(1)
base_info['person_NULL_SUM'] = base_info[['empnum', 'parnum', 'exenum']].isnull().astype(int).sum(1)

# base_info['regcap_DIVDE_empnum'] = base_info['regcap'] / base_info['empnum']
# base_info['regcap_DIVDE_exenum'] = base_info['regcap'] / base_info['exenum']

# base_info['reccap_DIVDE_empnum'] = base_info['reccap'] / base_info['empnum']
# base_info['regcap_DIVDE_exenum'] = base_info['regcap'] / base_info['exenum']

# base_info['congro_DIVDE_empnum'] = base_info['congro'] / base_info['empnum']
# base_info['regcap_DIVDE_exenum'] = base_info['regcap'] / base_info['exenum']

base_info['opfrom'] = pd.to_datetime(base_info['opfrom'])
base_info['opto'] = pd.to_datetime(base_info['opto'])
base_info['opfrom_TONOW'] = (datetime.datetime.now() - base_info['opfrom']).dt.days
base_info['opfrom_TIME'] = (base_info['opto'] - base_info['opfrom']).dt.days

base_info['opscope_COUNT'] = base_info['opscope'].apply(
    lambda x: len(x.replace("\t", "，").replace("\n", "，").split('、')))

cat_col = ['oplocdistrict', 'industryphy', 'industryco', 'enttype',
           'enttypeitem', 'enttypeminu', 'enttypegb',
           'dom', 'oploc', 'opform']

for col in cat_col:
    base_info[col + '_COUNT'] = base_info[col].map(base_info[col].value_counts())
    col_idx = base_info[col].value_counts()
    for idx in col_idx[col_idx < 10].index:
        base_info[col] = base_info[col].replace(idx, -1)

# base_info['opscope'] = base_info['opscope'].apply(lambda x: x.replace("\t", " ").replace("\n", " ").replace("，", " "))
# clf_tfidf = TfidfVectorizer(max_features=200)
# tfidf=clf_tfidf.fit_transform(base_info['opscope'])
# tfidf = pd.DataFrame(tfidf.toarray())
# tfidf.columns = ['opscope_' + str(x) for x in range(200)]
# base_info = pd.concat([base_info, tfidf], axis=1)

base_info = base_info.drop(['opfrom', 'opto'], axis=1)

for col in ['industryphy', 'dom', 'opform', 'oploc']:
    base_info[col] = pd.factorize(base_info[col])[0]

report = report.drop(drop_col_by_nan(report, 0.01), axis=1)
report_df = report.groupby('id').agg({
    'ANCHEYEAR': ['max'],
    'STATE': ['max'],
    'FUNDAM': ['max'],
    'EMPNUM': ['max'],
    'UNEEMPLNUM': ['max', 'sum']
})
report_df.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper()
                              for e in report_df.columns.tolist()])
report_df = report_df.reset_index()

# change = change.drop(drop_col_by_nan(change, 0.01), axis=1)
change['bgrq'] = (change['bgrq'] / 10000000000).astype(int)
change_df = change.groupby('id').agg({
    'bgxmdm': ['count', 'nunique'],
    'bgq': ['nunique'],
    'bgh': ['nunique'],
    'bgrq': ['nunique'],
})
change_df.columns = pd.Index(['changeinfo_' + e[0] + "_" + e[1].upper()
                              for e in change_df.columns.tolist()])
change_df = change_df.reset_index()

# news = news.drop(drop_col_by_nan(news, 0.01), axis=1)
news['public_date'] = news['public_date'].apply(lambda x: x if '-' in str(x) else np.nan)
news['public_date'] = pd.to_datetime(news['public_date'])
news['public_date'] = (datetime.datetime.now() - news['public_date']).dt.days

news_df = news.groupby('id').agg({'public_date': ['count', 'max', 'min', 'mean']}).reset_index()
news_df.columns = ['id', 'public_date_COUNT', 'public_MAX', 'public_MIN', 'public_MEAN']
news_df1 = pd.pivot_table(news, index='id', columns='positive_negtive', aggfunc='count').reset_index()
news_df1.columns = ['id', 'news_COUNT1', 'news_COUNT2', 'news_COUNT3']
news = pd.merge(news_df, news_df1)

# tax = tax.drop(drop_col_by_nan(tax, 0.01), axis=1)
tax_df = tax.groupby('id').agg({
    'TAX_CATEGORIES': ['count'],
    'TAX_ITEMS': ['count'],
    'TAXATION_BASIS': ['count'],
    'TAX_AMOUNT': ['max', 'min', 'mean'],
})
tax_df.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper()
                           for e in tax_df.columns.tolist()])
tax_df = tax_df.reset_index()

# other = other.drop(drop_col_by_nan(other, 0.01), axis=1)
other = other[~other['id'].duplicated(keep='last')]
other['other_SUM'] = other[['legal_judgment_num', 'brand_num', 'patent_num']].sum(1)
other['other_NULL_SUM'] = other[['legal_judgment_num', 'brand_num', 'patent_num']].isnull().astype(int).sum(1)

# 训练集和测试集
train_data = pd.merge(base_info, label, on='id')
train_data = pd.merge(train_data, other, on='id', how='left')

train_data = pd.merge(train_data, news_df, on='id', how='left')
train_data = pd.merge(train_data, tax_df, on='id', how='left')
train_data = pd.merge(train_data, report_df, on='id', how='left')
train_data = pd.merge(train_data, change_df, on='id', how='left')

entprise_evaluate = entprise_evaluate[['id']]
test_data = pd.merge(base_info, entprise_evaluate, on='id')
test_data = pd.merge(test_data, other, on='id', how='left')
test_data = pd.merge(test_data, news_df, on='id', how='left')
test_data = pd.merge(test_data, tax_df, on='id', how='left')
test_data = pd.merge(test_data, report_df, on='id', how='left')
test_data = pd.merge(test_data, change_df, on='id', how='left')

pd.set_option('display.min_rows', 1000, 'display.max_rows', 1000)


# 评价分数函数
def eval_score(y_test, y_pre):
    _, _, f_class, _ = precision_recall_fscore_support(y_true=y_test, y_pred=y_pre, labels=[0, 1], average=None)
    fper_class = {'合法': f_class[0], '违法': f_class[1], 'f1': f1_score(y_test, y_pre)}
    return fper_class


def k_fold_searchParameters(model, train_val_data, train_val_kind, test_kind):
    mean_f1 = 0
    mean_f1Train = 0
    n_splits = 5

    cat_features = ['oplocdistrict', 'industryphy', 'industryco', 'enttype',
                    'enttypeitem', 'enttypeminu', 'enttypegb',
                    'dom', 'oploc', 'opform']
    sk = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2020)
    pred_test = np.zeros(len(test_kind))
    for train, test in sk.split(train_val_data, train_val_kind):
        x_train = train_val_data.iloc[train]
        y_train = train_val_kind.iloc[train]
        x_test = train_val_data.iloc[test]
        y_test = train_val_kind.iloc[test]
        # train_pool = Pool(x_train, y_train, cat_features=cat_features)
        # validate_pool = Pool(x_test, y_train, cat_features=cat_features)
        # model.fit(train_pool, eval_set=validate_pool,
        #           early_stopping_rounds=100,
        #           verbose=False)
        model.fit(x_train, y_train, eval_set=[(x_test, y_test)],
                  categorical_feature=cat_features,
                  early_stopping_rounds=100,
                  verbose=False)
        # 验证集
        predict_test_res = model.predict(x_test)
        fper_class_test = eval_score(y_test, predict_test_res)

        # 训练集
        predict_train_res = model.predict(x_train)
        fper_class_train = eval_score(y_train, predict_train_res)

        # 测试集
        pred_test += model.predict_proba(test_kind)[:, 1] / n_splits

        mean_f1 += fper_class_test['f1'] / n_splits
        mean_f1Train += fper_class_train['f1'] / n_splits
    return mean_f1, pred_test


train_after_drop = train_data.drop(['id', 'opscope', 'label'], axis=1)
train_data_label = train_data['label']
test_after_drop = test_data.drop(['id', 'opscope'], axis=1)

train_cols = train_after_drop.columns
test_cols = test_after_drop.columns

for i in train_after_drop.columns:
    if train_after_drop[i].dtypes == np.float:
        train_after_drop[i] = train_after_drop[i].astype('str')
for i in test_after_drop.columns:
    if test_after_drop[i].dtypes == np.float:
        test_after_drop[i] = test_after_drop[i].astype('str')
categorical_features_indices = np.where(train_after_drop.dtypes != np.float)[0]
x_train, x_validation, y_train, y_validation = train_test_split(train_after_drop, train_data_label, train_size=0.75,
                                                                random_state=2020)
train_pool = Pool(x_train, y_train, cat_features=categorical_features_indices)
validate_pool = Pool(x_validation, y_validation, cat_features=categorical_features_indices)
# cross validation
params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'eval_metric': 'F1',
    'random_seed': 42,
    'max_depth': 5,
    'logging_level': 'Silent',
    'use_best_model': True
}
# best_model catboost
best_model = cb.CatBoostClassifier(**params)

# print(type(test_label))
# print(test_label.shape)

# 预测times次数，取均值
times = 20
test_label = None
score_list = []
for i in range(times):
    best_model.fit(train_pool, eval_set=validate_pool)
    score_list.append(f1_score(y_validation, best_model.predict(x_validation)))
    if i == 0:
        test_label = best_model.predict_proba(test_after_drop)[:, 1]
    else:
        test_label += best_model.predict_proba((test_after_drop))[:, 1]
test_label = test_label / times
print(score_list)
print('Best model f1: {:.4}'.format(np.array(score_list).mean()))
print(np.sum(test_label < 0.5))

# # 网格搜索优化参数
# params_optimize = {
#     "learning_rate": [0.03, 0.05],
#     "max_depth": [5, 6, 7],
#     "random_seed": [40, 50, 60]
# }
# grid_search = GridSearchCV(best_model, n_jobs=-1, param_grid=params_optimize, cv=5, scoring='f1', verbose=5)
# grid_search.fit(x_train, y_train)
# print("best", grid_search.best_estimator_)
# print("best_score: ", grid_search.best_score_)
# print("best_params: ", grid_search.best_params_)

# # hyperopt metric optimize
# space = {
#     'iterations': hp.randint('iterations', 1000),
#     'random_seed': hp.randint('random_seed', 100),
#     # 'learning_rate': hp.choice('learning_rate', [0.02, 0.05, 0.1]),
#     # 'depth': hp.randint('max_depth', 11)
#
# }
#
#
# def hyperopt_train_test(params):
#     X_ = x_train[:]
#     if 'normalize' in params:
#         if params['normalize'] == 1:
#             X_ = normalize(X_)
#         del params['normalize']
#     if 'scale' in params:
#         if params['scale'] == 1:
#             X_ = scale(X_)
#         del params['scale']
#     clf = best_model
#     # recall = cross_val_score(clf, x_train, y_train, scoring='recall').mean()
#     f1 = cross_val_score(clf, x_train, y_train, scoring='f1').mean()
#     # precision = cross_val_score(clf, x_train, y_train, scoring='f1').mean()
#     return f1
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
#     max_evals=30,
#     trials=trials
# )
# print("best:  ", end='')
# print(best)
# for trial in trials.trials:
#     print(trial['result'])

# model = lgb.LGBMClassifier(learning_rate=0.05, max_depth=7, n_estimators=150,
#                            num_leaves=7, silent=False, n_jobs=-1)
#
# # lgb训练模型
# train_times = 20
# score_tta = None
# score_list = []
# for _ in range(train_times):
#     clf = model
#     score, test_pred = k_fold_searchParameters(clf, train_data.drop(['id', 'opscope', 'label'], axis=1),
#                                                train_data['label'],
#                                                test_data.drop(['id', 'opscope'], axis=1)
#                                                )
#     if score_tta is None:
#         score_tta = test_pred / train_times
#     else:
#         score_tta += test_pred / train_times
#     score_list.append(score)
#
# print("*********" * 8)
# print(np.array(score_list).mean(), np.array(score_list).std())


code_end = datetime.datetime.now()
print("code run time: ", code_end - code_start)

# 写入测试结果
test_data['score'] = test_label
test_data[['id', 'score']].to_csv('./test_label_folder/all_datafile_catboost20_depth5_1202.csv', sep=',', header=True,
                                  index=False)

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


# print('Best model validation accuracy: {:.4}'.format(
#     f1_score(y_validation, best_model.predict(x_validation))
# ))


# # 输出每个特征值的权重
# importances = best_model.get_feature_importance(prettified=True)
# print(importances)
#
# # 输出验证集的F1分数
# metrics = best_model.eval_metrics(data=train_pool, metrics=['Logloss', 'F1', 'Precision', 'Recall'])
# print("Logloss: ", min(metrics['Logloss']))
# print("f1 score: ", max(metrics['F1']))
# print("Precision: ", max(metrics['Precision']))
# print("Recall: ", max(metrics['Recall']))


# # 将预测结果保存并写入文件
# final_test_set['label'] = test_label
#
# # 需要提交的顺序
# test_submit = pd.read_csv(test_file_name)
# del test_submit['score']
#
# # 合并提交文件和预测结果
# test_submit = pd.merge(test_submit, final_test_set, on='id', how='left')
# test_submit.rename(columns={'label': 'score'}, inplace=True)
#
# # 写入文件
# save_test_res_name = './test_label_folder/base_report_fit_no_fill_0.95train_1124.csv'
# test_submit.to_csv(save_test_res_name, sep=',', header=True, index=False)
