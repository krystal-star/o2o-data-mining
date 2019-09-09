# model_xgboost.py

import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def model_xgb(train, test):
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'silent': 1,
              'eta': 0.01,
              'max_depth': 5,
              'min_child_weight': 1.1,
              'gamma': 0.1,
              'lambda': 10,
              'colsample_bylevel': 0.7,
              'colsample_bytree': 0.7,
              'subsample': 0.7,
              'scale_pos_weight': 1}

    datatrain = xgb.DMatrix(train.drop(['User_id', 'Coupon_id', 'Date_received','label'], axis=1), label=train['label'])
    datatest = xgb.DMatrix(test.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1))

    # 训练
    watchlist = [(datatrain, 'train')]
    model = xgb.train(params, datatrain, num_boost_round=1000,evals=watchlist)
    # 预测
    p = model.predict(datatest)
    # 结果处理
    p = pd.DataFrame(p, columns=['prob'])
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], p], axis=1)
    # 归一化
    result['prob'] = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(
        result['prob'].values.reshape(-1, 1))

    return result


print('读取文件ing')
train = pd.read_csv('/Users/liukai/PycharmProjects/datalab/traintest.csv')
validate = pd.read_csv('/Users/liukai/PycharmProjects/datalab/validatetest.csv')
test = pd.read_csv('/Users/liukai/PycharmProjects/datalab/testtest.csv')

train.drop_duplicates(inplace=True)
validate.drop_duplicates(inplace=True)
test.drop_duplicates(inplace=True)
x_train = pd.concat([train, validate], axis=0)
x_train = x_train.drop(['weekday3.1','weekday2.1','day_of_week', 'Date', 'weekday8.1','is_weekend.1','weekday5', 'weekday4'
                           , 'weekday6.1','weekday5.1','weekday7', 'weekday3', 'weekday6', 'weekday1', 'day_of_month'
                           , 'weekday2', 'weekday8', 'weekday1.1', 'is_weekend', 'Date.1','day_of_week.1','day_of_month.1',
                        'weekday4.1','weekday7.1'],axis=1)

# 训练
print('训练开始')
result = model_xgb(x_train, test)
result.to_csv('base.csv', index=False, header=None)

# 基于线性加权的模型融合
# 对特征进行抽样
y_train = x_train.drop(['User_id','Coupon_id','Date_received','label'],axis=1)
y_train_chouyang = y_train.sample(frac=0.7,axis=1,random_state=2000)

# 拼接前面四项即可得到抽样的训练集
y_train_chouyang = pd.concat([x_train[['User_id','Coupon_id','Date_received','label']],y_train_chouyang],axis=1)
# 提取训练集列名构成
columns = y_train_chouyang.columns.tolist()
columns.remove('label')
# 得到测试集列名构成
test_chouyang = test[columns]
# 对采样的训练集和测试集进行学习
print('融合开始')
result_chouyang = model_xgb(y_train_chouyang,test_chouyang)
result_chouyang.to_csv('chouyang.csv',index=False,header=None)
# 把前面两个预测结果进行融合
# 对前后两个模型的预测概率重命名
ronghe = pd.concat([result.rename(columns={'prob':'prob_all'}),
                             result_chouyang[['prob']].rename(columns={'prob':'prob_sample'})],axis=1)

ronghe['prob'] = ronghe['prob_all']*0.7 + ronghe['prob_sample']*0.3
ronghe[['User_id','Coupon_id','Date_received','prob']].to_csv('ronghe.csv',index=False,header=None)

