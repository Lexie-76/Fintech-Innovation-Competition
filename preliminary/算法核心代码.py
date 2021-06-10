import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from hyperopt import hp
import hyperopt
import warnings
warnings.filterwarnings("ignore")

# 处理空数据较多的列
def process_NaN(data):
    temp = []  # 临时存放列表
    for col in data.columns:
        count = data[col].isnull().sum()  # 每列空数据求和
        ratio = count / len(data[col])  # 空数据占该列所有数据的比例
        if ratio > 0.8:  # 0.9可以调整，观察对结果的影响情况
            # print(col)
            temp.append(col)  # 将符合条件的列放入临时列表中

    new_data = data.drop(temp, axis=1)  # 去掉空数据较多的列后的新数据
    return new_data

# 获取重复数据较多的列
def get_more_duplicate(data):
    new_cols = [col for col in data.columns
                if data[col].value_counts(dropna=False, normalize=True).values[0] > 0.99]  # 同一数据出现在某一列的概率大于0.99
    return new_cols

# 处理重复数据较多的列
def process_duplicate(train, test):
    train_cols = get_more_duplicate(train)  # train中重复数据较多的列
    test_cols = get_more_duplicate(test)  # test中重复数据较多的列
    union_cols = list(set(train_cols + test_cols))  # 二者的并集
    train = train.drop(union_cols, axis=1)
    test = test.drop(union_cols,axis=1)
    return train, test

# 比较
def compare(train, test):
    temp = []
    for col in train.columns:
        if col not in test.columns:
            temp.append(col)
    return temp

# 使用LabelEncoder处理分类变量
def label_encoder(train, test):
    for col in train.columns:
        if train[col].dtype == 'object':
            encoder = LabelEncoder()
            encoder.fit(list(train[col].values) + list(test[col].values))  # 训练集和测试集中的对应列
            train[col] = encoder.transform(list(train[col].values))  # 训练集中的object类转为int
            test[col] = encoder.transform(list(test[col].values))
    train = train.reset_index()  # 重置数据集
    test = test.reset_index()
    return train, test

# hyperopt目标函数
def objective(params):
    # 基线模型
    model = lgb.LGBMClassifier(objective='multiclass',
                               num_class=8,
                               max_depth=int(params['max_depth']) + 10,
                               metric='multi_logloss',
                               learning_rate=params['learning_rate'],
                               random_state=66)
    hyperparams = model.get_params()  # 获取基线模型的参数
    del hyperparams['n_estimators']  # 删除'n_estimators'
    # 交叉验证
    cv_results = lgb.cv(hyperparams,
                        train_dataset,
                        num_boost_round=10000,
                        nfold=5,
                        metrics='multi_logloss',
                        early_stopping_rounds=200,
                        verbose_eval=False,
                        seed=42)
    return min(cv_results['multi_logloss-mean'])




if __name__ == '__main__':
# 训练集和测试集
    train_data = pd.read_csv('train_new.csv')
    test_data = pd.read_csv('test_new.csv')
# 清洗数据
    train_data = process_NaN(train_data)  # 空数据处理
    test_data = process_NaN(test_data)
    # duplicate处理
    train_data, test_data = process_duplicate(train_data, test_data)
# 分类数据处理
    train_data, test_data = label_encoder(train_data, test_data)
# 特征和标签
    X = train_data.drop(['Response', 'Id'], axis=1)  # 用于训练的数据集
    y = train_data['Response']  # 用于训练的标签
    y = [x-1 for x in y]  # 8个度量序数为0-7
    test_Id = test_data['Id']  # Id列单独保存
    X_test = test_data.drop(['Id'], axis=1)  # 用于测试的数据集
# 划分训练集和验证集
    X_train_data, X_valid_data, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 42)
    train_dataset = lgb.Dataset(X_train_data, label=y_train)  # 训练集
    valid_dataset = lgb.Dataset(X_valid_data, label=y_valid)  # 验证集

# 搜索空间
    space = {
        'num_leaves': hp.quniform('num_leaves', 20, 150, 1),
        'max_depth': hyperopt.hp.randint('max_depth', 15),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.5)),
        'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
        'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
        'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
        'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
        'is_unbalance': hp.choice('is_unbalance', [True, False]),
    }
    trials = hyperopt.Trials()
# 最优结果
    best = hyperopt.fmin(
        objective,
        space=space,
        algo=hyperopt.tpe.suggest,
        max_evals=100,
        trials=trials
    )

    bestParams={
        'boosting_type': 'gbdt',
        'silent': True,
        'objective': 'multiclass',
        'metric':'multi_logloss',
        'learning_rate': best['learning_rate'],
        'num_leaves': int(best['num_leaves']),
        'max_depth': int(best['max_depth']),
        'colsample_bytree': best['colsample_bytree'],
        'reg_alpha': best['reg_alpha'],
        'reg_lambda': best['reg_lambda'],
        'min_child_samples': int(best['min_child_samples']),
        'num_class': 8
    }

# 预测
    model = lgb.LGBMClassifier(**bestParams)
    model.fit(X_train_data,y_train)
    y_pred = model.predict(X_test)
    y_pred = [x+1 for x in y_pred]
#输出
    result={
        'id':test_Id.values,
        'target':y_pred
    }
    df = pd.DataFrame(result)
    df.to_csv("C:\\Users\\Orchidea\\Desktop\\result.csv",index=False)


