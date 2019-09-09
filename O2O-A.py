# feature.py

import pandas as pd
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from feature_project import history_feature,middle_feature,label_feature
import warnings
warnings.filterwarnings('ignore')


def get_dataset(hf, mf, lf):
    # 特征工程
    print('开始构造数据集')
    h_feat = history_feature(lf, hf)
    h_feat_x = h_feat.drop(['User_id', 'label', 'Coupon_id','Discount_rate'], axis=1)
    for i in h_feat_x.columns:
        h_feat_x[i] = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(h_feat_x[i].values.reshape(-1, 1))
    h_feat_norm = pd.concat([h_feat['User_id'], h_feat['label'], h_feat['Coupon_id'], h_feat['Discount_rate'],h_feat_x], axis=1)

    m_feat = middle_feature(lf, mf)
    m_feat_x = m_feat.drop(['User_id', 'label', 'Coupon_id', 'Discount_rate','Date_received_y'], axis=1)
    for i in m_feat_x.columns:
        m_feat_x[i] = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(m_feat_x[i].values.reshape(-1, 1))
    m_feat_norm = pd.concat([m_feat['User_id'], m_feat['label'], m_feat['Coupon_id'],m_feat['Discount_rate'], m_feat_x], axis=1)

    l_feat = label_feature(lf)
    l_feat_x = l_feat.drop(['User_id','Coupon_id', 'Discount_rate','Merchant_id'], axis=1)
    for i in l_feat_x.columns:
        l_feat_x[i] = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(l_feat_x[i].values.reshape(-1, 1))
    l_feat_norm = pd.concat([l_feat['User_id'], l_feat['Coupon_id'],l_feat['Merchant_id'],l_feat['Discount_rate'], l_feat_x], axis=1)

    label = m_feat['label'].values.tolist()

    # 构造数据集
    print('构造数据集')
    share = list(set(h_feat_norm.columns.tolist()) & set(m_feat_norm.columns.tolist()) &
                            set(l_feat_norm.columns.tolist()))
    dataset = pd.concat([h_feat_norm, m_feat_norm.drop(share, axis=1)], axis=1)
    dataset = pd.concat([dataset, l_feat_norm.drop(share, axis=1)], axis=1)

    dataset.drop(['Merchant_id', 'date_received','date','label','Discount_rate'], axis=1, inplace=True)
    dataset['label'] = label

    # 修正数据类型
    dataset['User_id'] = dataset['User_id'].fillna(0).astype(int)
    dataset['Coupon_id'] = dataset['Coupon_id'].fillna(0).astype(int)
    dataset['Date_received'] = dataset['Date_received'].fillna(0).astype(int)
    dataset['Distance'] = dataset['Distance'].fillna(0).astype(int)
    dataset['label'] = dataset['label'].fillna(0).astype(int)

    return dataset


def get_test_dataset(hf, mf, lf):
    # 特征工程
    print('开始构造测试集')
    h_feat = history_feature(lf, hf)
    h_feat_x = h_feat.drop(['User_id', 'Coupon_id', 'Discount_rate'], axis=1)
    for i in h_feat_x.columns:
        h_feat_x[i] = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(h_feat_x[i].values.reshape(-1, 1))
    h_feat_norm = pd.concat([h_feat['User_id'], h_feat['Discount_rate'], h_feat['Coupon_id'], h_feat_x], axis=1)

    m_feat = middle_feature(lf, mf)
    m_feat_x = m_feat.drop(['User_id', 'Coupon_id', 'Discount_rate', 'Date_received_y'], axis=1)
    for i in m_feat_x.columns:
        m_feat_x[i] = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(m_feat_x[i].values.reshape(-1, 1))
    m_feat_norm = pd.concat([m_feat['User_id'], m_feat['Discount_rate'], m_feat['Coupon_id'], m_feat_x], axis=1)

    l_feat = label_feature(lf)
    l_feat_x = l_feat.drop(['User_id', 'Coupon_id', 'Discount_rate', 'Merchant_id'], axis=1)
    for i in l_feat_x.columns:
        l_feat_x[i] = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(l_feat_x[i].values.reshape(-1, 1))
    l_feat_norm = pd.concat([l_feat['User_id'], l_feat['Coupon_id'], l_feat['Merchant_id'],l_feat['Discount_rate'],l_feat_x], axis=1)

    # 构造数据集
    print('构造数据集')
    share = list(set(h_feat_norm.columns.tolist()) & set(m_feat_norm.columns.tolist()) &
                            set(l_feat_norm.columns.tolist()))
    dataset = pd.concat([h_feat_norm, m_feat_norm.drop(share, axis=1)], axis=1)
    dataset = pd.concat([dataset, l_feat_norm.drop(share, axis=1)], axis=1)

    dataset.drop(['Merchant_id', 'date_received','Discount_rate'], axis=1, inplace=True)

    # 修正数据类型
    dataset['User_id'] = dataset['User_id'].fillna(0).astype(int)
    dataset['Coupon_id'] = dataset['Coupon_id'].fillna(0).astype(int)
    dataset['Date_received'] = dataset['Date_received'].fillna(0).astype(int)
    dataset['Distance'] = dataset['Distance'].fillna(0).astype(int)

    return dataset


def get_label(dataset):
    data = dataset.copy()
    data['label'] = list(map(lambda x, y: 1 if (x - y).total_seconds() / (60 * 60 * 24) <= 15 else 0, data['date'],
                             data['date_received']))
    return data


def train_prepare(train):
    train['is_manjian'] = train['Discount_rate'].apply(lambda x: 1 if ':' in str(x) else 0)
    train['discount_rate'] = train['Discount_rate'].apply(lambda x: float(x) if ':' not in str(x) else
    (float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))
    train['min_cost_of_manjian'] = train['Discount_rate'].apply(
        lambda x: -1 if ':' not in str(x) else int(str(x).split(':')[0]))
    train['date_received'] = pd.to_datetime(train['Date_received'], format='%Y%m%d')
    train['date'] = pd.to_datetime(train['Date'], format='%Y%m%d')

    train['Distance'].fillna(-1, inplace=True)
    train['null_distance'] = train['Distance'].apply(lambda x: 1 if x == -1 else 0)

    # 星期几
    train['day_of_week'] = train['Date_received'].astype('str').apply(
        lambda x: date(int(x[0:4]), int(x[4:6]), int(x[6:8])).weekday() + 1 if x != 'nan'else 0)
    # 几月
    train['day_of_month'] = train['Date_received'].astype('str').apply(lambda x: int(x[6:8]) if x != 'nan'else 0)
    # 是否周末
    train['is_weekend'] = train['day_of_week'].apply(lambda x: 1 if x in (6, 7) else 0)
    # 将day_of_week序列化，化为7列01量
    weekday_dummies = pd.get_dummies(train['day_of_week'])
    weekday_dummies.columns = [
        'weekday' + str(i + 1) for i in range(weekday_dummies.shape[1])]
    # 合并
    train = pd.concat([train, weekday_dummies], axis=1)

    return train


def test_prepare(train):
    train['is_manjian'] = train['Discount_rate'].apply(lambda x: 1 if ':' in str(x) else 0)
    train['discount_rate'] = train['Discount_rate'].apply(lambda x: float(x) if ':' not in str(x) else
    (float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))
    train['min_cost_of_manjian'] = train['Discount_rate'].apply(
        lambda x: -1 if ':' not in str(x) else int(str(x).split(':')[0]))
    train['date_received'] = pd.to_datetime(train['Date_received'], format='%Y%m%d')

    train['Distance'].fillna(-1, inplace=True)
    train['null_distance'] = train['Distance'].apply(lambda x: 1 if x == -1 else 0)

    return train


# main
print('读取文件ing')
o_train = pd.read_csv('/Users/liukai/Desktop/datamining/tianchi_data/ccf_offline_stage1_train.csv')
o_test = pd.read_csv('/Users/liukai/Desktop/datamining/tianchi_data/ccf_offline_stage1_test_revised.csv')

o_train = train_prepare(o_train)
o_test = test_prepare(o_test)

print('打标ing')
o_train = get_label(o_train)

# 划分区间
# 训练集历史区间、中间区间、标签区间
train_hf = o_train[o_train['date_received'].isin(pd.date_range('2016/3/2', periods=60))]
train_mf = o_train[o_train['date'].isin(pd.date_range('2016/5/1', periods=15))]
train_lf = o_train[o_train['date_received'].isin(pd.date_range('2016/5/16', periods=31))]

# 验证集历史区间、中间区间、标签区间
validate_hf = o_train[o_train['date_received'].isin(pd.date_range('2016/1/16', periods=60))]
validate_mf = o_train[o_train['date'].isin(pd.date_range('2016/3/16', periods=15))]
validate_lf = o_train[o_train['date_received'].isin(pd.date_range('2016/3/31', periods=31))]

# 测试集历史区间、中间区间、标签区间
test_hf = o_train[o_train['date_received'].isin(pd.date_range('2016/4/17', periods=60))]
test_mf = o_train[o_train['date'].isin(pd.date_range('2016/6/16', periods=15))]
test_lf = o_test.copy()

trainset = get_dataset(train_hf,train_mf,train_lf)
trainset.to_csv('traintest.csv',index=None)
validateset = get_dataset(validate_hf,validate_mf,validate_lf)
validateset.to_csv('validatetest.csv',index=None)
testset = get_test_dataset(test_hf,test_mf,test_lf)
testset.to_csv('testtest.csv',index=None)
