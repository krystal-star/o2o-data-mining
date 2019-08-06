import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import hf
import lf
import mf


def get_dataset(history_field, middle_field, label_feild):
    # 特征工程
    history_feat = hf.get_history_field_feature(label_feild, history_field)
    middle_feat = mf.get_middle_field_feature(label_feild, middle_field)
    label_feat = lf.get_label_field_feature(label_feild)

    # 构造数据集
    share_characters = list(set(history_feat.columns.tolist()) & set(middle_feat.columns.tolist()) &
                            set(label_feat.columns.tolist()))
    dataset = pd.concat([history_feat, middle_feat.drop(share_characters, axis=1)], axis=1)
    dataset = pd.concat([dataset, label_feat.drop(share_characters, axis=1)], axis=1)

    # 删除无用属性并将label置于最后一列
    if 'Date' in dataset.columns.tolist(): # 训练集和验证集
        dataset.drop(['Merchant_id', 'Discount_rate', 'date_received', 'Date', 'date'], axis=1, inplace=True)
        label = dataset['label'].values.tolist()
        dataset.drop(['label'], axis=1, inplace=True)
        dataset['label'] = label
    else:    # 测试集
        dataset.drop(['Merchant_id', 'Discount_rate', 'date_received'], axis=1, inplace=True)


    # 修正数据类型
    dataset['User_id'] = dataset['User_id'].fillna(0).astype(int)
    dataset['Coupon_id'] = dataset['Coupon_id'].fillna(0).astype(int)
    dataset['Date_received'] = dataset[['Date_received']].fillna(0).astype(int)
    dataset['Distance'] = dataset[['Distance']].fillna(0).astype(int)
    dataset['label'] = dataset['label'].astype(int)

    return dataset


def get_test_dataset(history_field, middle_field, label_feild):
    # 特征工程
    history_feat = hf.get_history_field_feature(label_feild, history_field)
    middle_feat = mf.get_middle_field_feature(label_feild, middle_field)
    label_feat = lf.get_label_field_feature(label_feild)

    # 构造数据集
    share_characters = list(set(history_feat.columns.tolist()) & set(middle_feat.columns.tolist()) &
                            set(label_feat.columns.tolist()))
    dataset = pd.concat([history_feat, middle_feat.drop(share_characters, axis=1)], axis=1)
    dataset = pd.concat([dataset, label_feat.drop(share_characters, axis=1)], axis=1)
    dataset.drop(['Merchant_id', 'Discount_rate', 'date_received'], axis=1, inplace=True)

    # 修正数据类型
    dataset['User_id'] = dataset['User_id'].fillna(0).astype(int)
    dataset['Coupon_id'] = dataset['Coupon_id'].fillna(0).astype(int)
    dataset['Date_received'] = dataset[['Date_received']].fillna(0).astype(int)
    dataset['Distance'] = dataset[['Distance']].fillna(0).astype(int)

    return dataset


def get_label(dataset):
    data = dataset.copy()
    data['label'] = list(map(lambda x, y: 1 if (x - y).total_seconds() / (60 * 60 * 24) <= 15 else 0, data['date'],
                             data['date_received']))
    return data


def train_prepare(train):
    train['is_manjian'] = train['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)
    train['discount_rate'] = train['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else
    (float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))
    train['min_cost_of_manjian'] = train['Discount_rate'].map(lambda x:-1 if ':' not in str(x) else int(str(x).split(':')[0]))
    train['date_received'] = pd.to_datetime(train['Date_received'], format='%Y%m%d')
    train['date'] = pd.to_datetime(train['Date'], format='%Y%m%d')

    train['Distance'].fillna(-1, inplace=True)
    train['null_distance'] = train['Distance'].map(lambda x: 1 if x == -1 else 0)

    return train


def test_prepare(train):
    train['is_manjian'] = train['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)
    train['discount_rate'] = train['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else
    (float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))
    train['min_cost_of_manjian'] = train['Discount_rate'].map(lambda x:-1 if ':' not in str(x) else int(str(x).split(':')[0]))
    train['date_received'] = pd.to_datetime(train['Date_received'], format='%Y%m%d')

    train['Distance'].fillna(-1, inplace=True)
    train['null_distance'] = train['Distance'].map(lambda x: 1 if x == -1 else 0)

    return train


def model_xgb(train, test):
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'silent': 1,
              'eta': 0.01,
              'max_depth': 5,
              'min_child_weight': 1,
              'gamma': 0,
              'lambda': 1,
              'colsample_bylevel': 0.7,
              'colsample_bytree': 0.7,
              'subsample': 0.9,
              'scale_pos_weight': 1}

    dtrain = xgb.DMatrix(train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1), label=train['label'])
    dtest = xgb.DMatrix(test.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1))

    # 训练
    watchlist = [(dtrain, 'train')]
    model = xgb.train(params, dtrain, num_boost_round=1000, evals=watchlist)
    # 预测
    predict = model.predict(dtest)
    # 结果处理
    predict = pd.DataFrame(predict, columns=['prob'])
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], predict], axis=1)
    # 特征重要性
    feat_importance = pd.DataFrame(columns=['feature_name', 'importance'])
    feat_importance['feature_name'] = model.get_score().keys()
    feat_importance['importance'] = model.get_score().values()
    feat_importance.sort_values(['importance'], ascending=False)

    return result, feat_importance


# 性能评价函数
def off_evaluate(validate, off_result):
    evaluate_data = pd.concat([validate[['Coupon_id', 'label']], off_result[['prob']]], axis=1)
    aucs = 0
    lens = 0
    for name, group in evaluate_data.groupby('Coupon_id'):
        if len(set(list(group['label']))) != 2:
            continue
        aucs += roc_auc_score(group['label'], group['prob'])
        lens += 1
    auc = aucs / lens
    return auc


# main
off_train = pd.read_csv('/Users/liukai/Desktop/datamining/tianchi_data/ccf_offline_stage1_train.csv')
off_test = pd.read_csv('/Users/liukai/Desktop/datamining/tianchi_data/ccf_offline_stage1_test_revised.csv')

off_train = train_prepare(off_train)
off_test = test_prepare(off_test)

off_train = get_label(off_train)

# 划分区间
# 训练集历史区间、中间区间、标签区间
train_history_field = off_train[off_train['date_received'].isin(pd.date_range('2016/3/2', periods=60))]
train_middle_field = off_train[off_train['date'].isin(pd.date_range('2016/5/1', periods=15))]
train_label_field = off_train[off_train['date_received'].isin(pd.date_range('2016/5/16', periods=31))]

# 验证集历史区间、中间区间、标签区间
validate_history_field = off_train[off_train['date_received'].isin(pd.date_range('2016/1/16', periods=60))]
validate_middle_field = off_train[off_train['date'].isin(pd.date_range('2016/3/16', periods=15))]
validate_label_field = off_train[off_train['date_received'].isin(pd.date_range('2016/3/31', periods=31))]

# 测试集历史区间、中间区间、标签区间
test_history_field = off_train[off_train['date_received'].isin(pd.date_range('2016/4/17', periods=60))]
test_middle_field = off_train[off_train['date'].isin(pd.date_range('2016/6/16', periods=15))]
test_label_field = off_test.copy()


print('构造训练集')
train = get_dataset(train_history_field, train_middle_field, train_label_field)
print('构造验证集')
validate = get_dataset(validate_history_field, validate_middle_field, validate_label_field)
print('构造测试集')
test = get_test_dataset(test_history_field, test_middle_field, test_label_field)

# 保存
train.to_csv('/Users/liukai/Desktop/datamining/tianchi_data/train.csv')
validate.to_csv('/Users/liukai/Desktop/datamining/tianchi_data/validate.csv')
test.to_csv('/Users/liukai/Desktop/datamining/tianchi_data/test.csv')

# 线下
# off_result, off_feat_importance = model_xgb(train, validate.drop(['label'], axis=1))
# print('auc = ', off_evaluate(validate, off_result))

# 线上训练
big_train = pd.concat([train, validate], axis=0)
result, feat_importance = model_xgb(big_train,test)

result.to_csv('base.csv', index=False, header=None)

# 抽样训练
big_train_feat = big_train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)
big_train_sample = big_train_feat.sample(frac=0.7, axis=1, random_state=2018)
big_train_sample = pd.concat([big_train[['User_id', 'Coupon_id', 'Date_received', 'label']], big_train_sample], axis=1)
sample_cols = big_train_sample.columns.tolist()
sample_cols.remove('label')
test_sample = test[sample_cols]
result_sample, feat_importance_sample = model_xgb(big_train_sample, test_sample)
result_sample.to_csv('sample.csv', index=False, header=None)

# 融合
result_ronghe = pd.concat([result.rename(columns = {'prob': 'prob_all'}),
                           result_sample[['prob']].rename(columns = {'prob': 'prob_sample'})], axis=1)

result_ronghe['prob'] = result_ronghe['prob_all'] * 0.9 + result_ronghe['prob_sample'] * 0.1
result_ronghe[['User_id', 'Coupon_id', 'Date_received',
               'prob']].to_csv('/Users/liukai/Desktop/datamining/tianchi_data/ronghe.csv', index=False, header=None)

