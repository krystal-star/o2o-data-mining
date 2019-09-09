# feature_project.py

import pandas as pd
import numpy as np


def history_user_feature(label_field, history_field):
    # 源数据
    data = history_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1

    # 主键
    keys = ['User_id']
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    u_feat = label_field[keys].drop_duplicates(keep='first')

    # 用户领券数
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户未核销数
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) == 'nan')], index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_not_consume_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户核销数
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'nan')], index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_and_consume_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户核销率
    u_feat[prefixs + 'receive_and_consume_rate'] = list(map(lambda x, y: x / y if y != 0 else 0,
                                                            u_feat[prefixs + 'receive_and_consume_cnt'],
                                                            u_feat[prefixs + 'receive_cnt']))
    # 用户领取了多少个不同商家的优惠券
    pivot = pd.pivot_table(data, index=keys, values='Merchant_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'Merchant_id': prefixs + 'receive_differ_Merchant_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户对领券商家的15天内的核销数
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='Merchant_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(
        columns={'Merchant_id': prefixs + 'receive_and_consume_differ_Merchant_cnt_15'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户对领券商家的15天内的核销率
    u_feat[prefixs + 'receive_and_consume_rate_15_differ_Merchant'] = list(map(lambda x, y: x / y if y != 0 else 0,
                                                                               u_feat[
                                                                                   prefixs + 'receive_and_consume_differ_Merchant_cnt_15'],
                                                                               u_feat[
                                                                                   prefixs + 'receive_differ_Merchant_cnt']))

    # 用户核销数/用户对领券商家15天内的核销数
    u_feat[prefixs + 'receive_and_consume_rate_15_differ_Merchant'] = list(map(lambda x, y: x / y if y != 0 else 0,
                                                                               u_feat[
                                                                                   prefixs + 'receive_and_consume_cnt'],
                                                                               u_feat[
                                                                                   prefixs + 'receive_and_consume_differ_Merchant_cnt_15']))

    # 用户15天内核销的最大距离
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='Distance',
                           aggfunc=lambda x: np.max([np.nan if i == -1 else i for i in x]))
    pivot = pd.DataFrame(pivot).rename(
        columns={'Distance': prefixs + 'receive_and_consume_15_max_Distance'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(-1, downcast='infer', inplace=True)

    # 用户15天内核销的最小距离
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='Distance',
                           aggfunc=lambda x: np.min([np.nan if i == -1 else i for i in x]))
    pivot = pd.DataFrame(pivot).rename(
        columns={'Distance': prefixs + 'receive_and_consume_15_min_Distance'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(-1, downcast='infer', inplace=True)

    # 用户15天内核销的平均距离
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='Distance',
                           aggfunc=lambda x: np.mean([np.nan if i == -1 else i for i in x]))
    pivot = pd.DataFrame(pivot).rename(
        columns={'Distance': prefixs + 'receive_and_consume_15_mean_Distance'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(-1, downcast='infer', inplace=True)

    # 用户领取的不同券的数量
    pivot = pd.pivot_table(data, index=keys, values='Coupon_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'Coupon_id': prefixs + 'receive_differ_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户15天内核销的不同券的数量
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='Coupon_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(
        columns={'Coupon_id': prefixs + 'receive_and_consume_15_differ_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户对不同券的15天内的核销率
    u_feat[prefixs + 'receive_and_consume_differ_rate_15'] = list(map(lambda x, y: x / y if y != 0 else 0,
                                                                      u_feat[
                                                                          prefixs + 'receive_and_consume_15_differ_cnt'],
                                                                      u_feat[prefixs + 'receive_differ_cnt']))

    # 用户15天内被核销的平均时间间隔
    tmp = data[data['label'] == 1]
    tmp['gap'] = (tmp['date'] - tmp['date_received']).map(lambda x: x.total_seconds() / (60 * 60 * 24))
    pivot = pd.pivot_table(tmp, index=keys, values='gap', aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={'gap': prefixs + 'consumed_mean_time_gap_15'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(-1, downcast='infer', inplace=True)

    # 用户15天内被核销的最小时间间隔
    pivot = pd.pivot_table(tmp, index=keys, values='gap', aggfunc=np.min)
    pivot = pd.DataFrame(pivot).rename(columns={'gap': prefixs + 'consumed_min_time_gap_15'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(-1, downcast='infer', inplace=True)

    # 用户15天内被核销的最小折扣率
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='discount_rate', aggfunc=np.min)
    pivot = pd.DataFrame(pivot).rename(
        columns={'discount_rate': prefixs + 'consumed_15_min_discount_rate'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户15天内被核销的最大折扣率
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='discount_rate', aggfunc=np.max)
    pivot = pd.DataFrame(pivot).rename(
        columns={'discount_rate': prefixs + 'consumed_15_max_discount_rate'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户15天内被核销的折扣率中位数
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='discount_rate', aggfunc=np.median)
    pivot = pd.DataFrame(pivot).rename(
        columns={'discount_rate': prefixs + 'consumed_15_median_discount_rate'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    return u_feat


def history_merchant_feature(label_field, history_field):
    # 源数
    data = history_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1

    # 主键
    keys = ['Merchant_id']
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    m_feat = label_field[keys].drop_duplicates(keep='first')

    # 商家的优惠券被领取次数
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)

    # 商家被不同客户领取次数：
    pivot = pd.pivot_table(data, index=keys, values='User_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'User_id': prefixs + 'received_differ_User_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)

    # 商家的券被核销的次数：
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'nan')], index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_and_consumed_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)

    # 商家的券的被核销率：
    m_feat[prefixs + 'received_and_consumed_rate'] = list(map(lambda x, y: x / y if y != 0 else 0,
                                                              m_feat[prefixs + 'received_and_consumed_cnt'],
                                                              m_feat[prefixs + 'received_cnt']))

    # 商家的券没被核销的次数：
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) == 'nan')], index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_not_consumed_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)

    # 商家提供的不同的优惠券数：
    pivot = pd.pivot_table(data, index=keys, values='Coupon_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'Coupon_id': prefixs + 'differ_Coupon_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)

    # 平均每个用户核销多少张商家的券
    m_feat[prefixs + 'per_User_consume_cnt'] = list(map(lambda x, y: x / y if y != 0 else 0,
                                                        m_feat[prefixs + 'received_and_consumed_cnt'],
                                                        m_feat[prefixs + 'received_differ_User_cnt']))

    # 商家提供的券平均被使用多少次
    m_feat[prefixs + 'per_Coupon_consume_cnt'] = list(map(lambda x, y: x / y if y != 0 else 0,
                                                          m_feat[prefixs + 'received_and_consumed_cnt'],
                                                          m_feat[prefixs + 'differ_Coupon_cnt']))

    # 商家15天内被核销的券种类数量
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='Coupon_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(
        columns={'Coupon_id': prefixs + 'received_and_consumed_15_differ_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)

    # 商家15天内核销的券距离用户的最大距离
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='Distance',
                           aggfunc=lambda x: np.max([np.nan if i == -1 else i for i in x]))
    pivot = pd.DataFrame(pivot).rename(
        columns={'Distance': prefixs + 'received_and_consumed_15_max_Distance'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(-1, downcast='infer', inplace=True)

    # 商家15天内核销的券距离用户的平均距离
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='Distance',
                           aggfunc=lambda x: np.mean([np.nan if i == -1 else i for i in x]))
    pivot = pd.DataFrame(pivot).rename(
        columns={'Distance': prefixs + 'received_and_consumed_15_mean_Distance'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(-1, downcast='infer', inplace=True)

    # 商家15天内核销的券距离用户的最小距离
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='Distance',
                           aggfunc=lambda x: np.min([np.nan if i == -1 else i for i in x]))
    pivot = pd.DataFrame(pivot).rename(
        columns={'Distance': prefixs + 'received_and_consumed_15_min_Distance'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(-1, downcast='infer', inplace=True)

    # 商家15天内核销的券的最大折扣率
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='discount_rate', aggfunc=np.max)
    pivot = pd.DataFrame(pivot).rename(
        columns={'discount_rate': prefixs + 'consumed_15_max_discount_rate'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)

    # 商家15天内核销的券的最小折扣率
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='discount_rate', aggfunc=np.min)
    pivot = pd.DataFrame(pivot).rename(
        columns={'discount_rate': prefixs + 'consumed_15_min_discount_rate'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)

    # 商家15天内核销的券的折扣率中位数
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='discount_rate', aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(
        columns={'discount_rate': prefixs + 'consumed_15_mean_discount_rate'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)

    # 商家15天内被核销的券的平均时间间隔
    tmp = data[data['label'] == 1]
    tmp['gap'] = (tmp['date'] - tmp['date_received']).map(lambda x: x.total_seconds() / (60 * 60 * 24))
    pivot = pd.pivot_table(tmp, index=keys, values='gap', aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={'gap': prefixs + 'consumed_mean_time_gap_15'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(-1, downcast='infer', inplace=True)

    # 商家15天内被核销的不同种类券占所有不同种类券的比例
    m_feat[prefixs + prefixs + 'received_and_consumed_differ_rate'] = list(map(lambda x, y: x / y if y != 0 else 0,
                                                        m_feat[prefixs + 'received_and_consumed_15_differ_cnt'],
                                                        m_feat[prefixs + 'differ_Coupon_cnt']))

    m_feat.fillna(0, downcast='infer', inplace=True)

    return m_feat


def history_coupon_feature(label_field, history_field):
    # 源数
    data = history_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1

    # 主键
    keys = ['Coupon_id']
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    c_feat = label_field[keys].drop_duplicates(keep='first')

    # 优惠券被领取次数
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_cnt'}).reset_index()
    c_feat = pd.merge(c_feat, pivot, on=keys, how='left')
    c_feat.fillna(0, downcast='infer', inplace=True)

    # 优惠券15天内被核销的次数：
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_and_consumed_cnt_15'}).reset_index()
    c_feat = pd.merge(c_feat, pivot, on=keys, how='left')
    c_feat.fillna(0, downcast='infer', inplace=True)

    # 优惠券15天内的核销率
    c_feat[prefixs + 'received_and_consumed_rate_15'] = list(map(lambda x, y: x / y if y != 0 else 0,
                                                                 c_feat[prefixs + 'received_and_consumed_cnt_15'],
                                                                 c_feat[prefixs + 'received_cnt']))

    # 优惠券15天内被核销的平均时间间隔
    tmp = data[data['label'] == 1]
    tmp['gap'] = (tmp['date'] - tmp['date_received']).map(lambda x: x.total_seconds() / (60 * 60 * 24))
    pivot = pd.pivot_table(tmp, index=keys, values='gap', aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={'gap': prefixs + 'consumed_mean_time_gap_15'}).reset_index()
    c_feat = pd.merge(c_feat, pivot, on=keys, how='left')
    c_feat.fillna(-1, downcast='infer', inplace=True)

    # 满减优惠券最低消费的中位数：
    pivot = pd.pivot_table(data[data['is_manjian'] == 1], index=keys, values='min_cost_of_manjian', aggfunc=np.median)
    pivot = pd.DataFrame(pivot).rename(
        columns={'min_cost_of_manjian': prefixs + 'median_of_min_cost_of_manjian'}).reset_index()
    c_feat = pd.merge(c_feat, pivot, on=keys, how='left')
    c_feat.fillna(0, downcast='infer', inplace=True)

    return c_feat


def history_user_merchant_feature(label_field, history_field):
    # 源数据
    data = history_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1

    # 主键
    keys = ['User_id', 'Merchant_id']
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    um_feat = label_field[keys].drop_duplicates(keep='first')

    # 该用户领取该商家的优惠券的数目
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_cnt'}).reset_index()
    um_feat = pd.merge(um_feat, pivot, on=keys, how='left')
    um_feat.fillna(0, downcast='infer', inplace=True)

    # 该用户核销该商家的优惠券的数目
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'nan')], index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_and_consume_cnt'}).reset_index()
    um_feat = pd.merge(um_feat, pivot, on=keys, how='left')
    um_feat.fillna(0, downcast='infer', inplace=True)

    # 该用户未核销该商家的优惠券的数目
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) == 'nan')], index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_not_consume_cnt'}).reset_index()
    um_feat = pd.merge(um_feat, pivot, on=keys, how='left')
    um_feat.fillna(0, downcast='infer', inplace=True)

    # 该用户核销该商家的优惠券的概率
    um_feat[prefixs + 'receive_and_consume_rate'] = list(map(lambda x, y: x / y if y != 0 else 0,
                                                                 um_feat[prefixs + 'receive_and_consume_cnt'],
                                                                 um_feat[prefixs + 'received_cnt']))
    um_feat.fillna(0, downcast='infer', inplace=True)

    return um_feat


def history_user_coupon_feature(label_field, history_field):
    # 源数据
    data = history_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1

    # 主键
    keys = ['User_id', 'Coupon_id']
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    uc_feat = label_field[keys].drop_duplicates(keep='first')

    # 该用户领取该商家的优惠券数目
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_cnt'}).reset_index()
    uc_feat = pd.merge(uc_feat, pivot, on=keys, how='left')
    uc_feat.fillna(0, downcast='infer', inplace=True)

    # 该用户核销该商家的优惠券数目
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'nan')], index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_and_consume_cnt'}).reset_index()
    uc_feat = pd.merge(uc_feat, pivot, on=keys, how='left')
    uc_feat.fillna(0, downcast='infer', inplace=True)

    # 该用户核销该商家的优惠券的概率
    uc_feat[prefixs + 'receive_and_consume_rate'] = list(map(lambda x, y: x / y if y != 0 else 0,
                                                             uc_feat[prefixs + 'receive_and_consume_cnt'],
                                                             uc_feat[prefixs + 'received_cnt']))
    uc_feat.fillna(0, downcast='infer', inplace=True)

    return uc_feat


def history_other_feature(label_field,history_field):
    data = history_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1

    keys = ['User_id', 'Merchant_id']
    feat = label_field[keys].drop_duplicates(keep='first')

    t = data[['User_id', 'Merchant_id', 'Date']].copy()
    t = t[t['Date'] != 'nan'][['User_id', 'Merchant_id']]
    t['user_merchant_buy_total'] = 1
    t = t.groupby(['User_id', 'Merchant_id']).agg('sum').reset_index()
    # 去重没有任何意义，因为agg(sum)的时候已经相当于去重了，reset_index之后也不会变回agg前的行数
    t.drop_duplicates(inplace=True)

    # 一个客户在一个商家浏览的次数（领过优惠券或者买过商品）
    t1 = data[['User_id', 'Merchant_id']]
    t1['user_merchant_any'] = 1
    t1 = t1.groupby(['User_id', 'Merchant_id']).agg('sum').reset_index()
    t1.drop_duplicates(inplace=True)

    keys1 = ['User_id']
    feat1 = label_field[keys1].drop_duplicates(keep='first')
    # 客户使用优惠券购买的次数
    t2 = data[(data['Date'] != 'nan') & (data['Coupon_id'] != 'nan')][['User_id']]
    t2['buy_use_coupon'] = 1
    t2 = t2.groupby('User_id').agg('sum').reset_index()

    # 客户购买任意商品的总次数
    t3 = data[data['Date'] != 'nan'][['User_id']]
    t3['buy_total'] = 1
    t3 = t3.groupby('User_id').agg('sum').reset_index()

    # 客户收到优惠券的总数
    t4 = data[data['Coupon_id'] != 'nan'][['User_id']]
    t4['coupon_received'] = 1
    t4 = t4.groupby('User_id').agg('sum').reset_index()

    keys2 = ['User_id','Merchant_id']
    feat2 = label_field[keys2].drop_duplicates(keep='first')
    # 卖出的商品
    t5 = data[data['Date'] != 'nan'][['Merchant_id']].copy()
    t5['total_sales'] = 1
    # 每个商品的销售数量
    t5 = t5.groupby('Merchant_id').agg('sum').reset_index()

    # 使用了优惠券消费的商品，正样本
    t6 = data[(data['Date'] != 'nan') & (
            data['Coupon_id'] != 'nan')][['Merchant_id']].copy()
    t6['sales_use_coupon'] = 1
    t6 = t6.groupby('Merchant_id').agg('sum').reset_index()

    t7 = data[['User_id', 'Coupon_id']].copy()
    t7['this_month_user_receive_same_coupn_count'] = 1
    t7 = t7.groupby(['User_id', 'Coupon_id']).agg('sum').reset_index()

    # t8:用户领取特定优惠券的最大时间和最小时间
    t8 = data[['User_id', 'Coupon_id', 'Date_received']].copy()
    t8['Date_received'] = t8['Date_received'].astype('str')
    t8 = t8.groupby(['User_id', 'Coupon_id'])['Date_received'].agg(
        lambda x: ':'.join(x)).reset_index()
    t8['receive_number'] = t8['Date_received'].apply(lambda s: len(s.split(':')))
    t8 = t8[t8['receive_number'] > 1]
    # 最大接受的日期
    t8['max_date_received'] = t8['Date_received'].apply(
        lambda s: max([int(d) for d in s.split(':')]))
    # 最小的接收日期
    t8['min_date_received'] = t8['Date_received'].apply(
        lambda s: min([int(d) for d in s.split(':')]))
    t8 = t8[['User_id', 'Coupon_id', 'max_date_received', 'min_date_received']]

    feat = pd.merge(feat, t, on=[
        'User_id', 'Merchant_id'], how='left')
    feat = pd.merge(feat, t1, on=[
        'User_id', 'Merchant_id'], how='left')
    feat1 = pd.merge(feat1, t2, on=[
        'User_id'], how='left')
    feat1 = pd.merge(feat1, t3, on=[
        'User_id'], how='left')
    feat1 = pd.merge(feat1, t4, on=[
        'User_id'], how='left')
    feat2 = pd.merge(
        feat2, t5, on='Merchant_id', how='left')
    feat2 = pd.merge(
        feat2, t6, on='Merchant_id', how='left')

    feat3 = data[['User_id', 'Coupon_id', 'Date_received']]
    feat3 = pd.merge(feat3, t7, on=['User_id', 'Coupon_id'], how='left')
    feat3 = pd.merge(feat3, t8, on=['User_id', 'Coupon_id'], how='left')
    feat3['this_month_user_receive_same_coupon_lastone'] = feat3['max_date_received'] - feat3['Date_received'].astype(int)
    feat3['this_month_user_receive_same_coupon_firstone'] = feat3['Date_received'].astype(
        int) - feat3['min_date_received']

    feat1['buy_use_coupon'] = feat1['buy_use_coupon'].replace(
        np.nan, 0)
    feat2['sales_use_coupon'] = feat2['sales_use_coupon'].replace(
        np.nan, 0)

    feat['user_merchant_rate'] = feat['user_merchant_buy_total'].astype(
        'float') / feat['user_merchant_any'].astype('float')
    feat1['buy_use_coupon_rate'] = feat1['buy_use_coupon'].astype('float') / feat1['buy_total'].astype(
        'float')
    feat1['user_coupon_transfer_rate'] = feat1['buy_use_coupon'].astype(
        'float') / feat1['coupon_received'].astype('float')
    feat2['coupon_rate'] = feat2['sales_use_coupon'].astype(
        'float') / feat2['total_sales']

    all_feat = pd.merge(feat1, feat, on=['User_id'], how='left')
    all_feat = pd.merge(all_feat, feat2, on=['User_id'], how='left')
    all_feat = pd.merge(all_feat, feat3, on=['User_id'], how='left')

    return all_feat


def history_feature(lf, hf):
    u_feat = history_user_feature(lf, hf)
    m_feat = history_merchant_feature(lf, hf)
    c_feat = history_coupon_feature(lf, hf)
    um_feat = history_user_merchant_feature(lf, hf)
    uc_feat = history_user_coupon_feature(lf, hf)
    o_feat = history_other_feature(lf,hf)

    # 添加特征
    feat = lf.copy()
    feat = pd.merge(feat, u_feat, on=['User_id'], how='left')
    feat = pd.merge(feat, m_feat, on=['Merchant_id'], how='left')
    feat = pd.merge(feat, c_feat, on=['Coupon_id'], how='left')
    feat = pd.merge(feat, um_feat, on=['User_id', 'Merchant_id'], how='left')
    feat = pd.merge(feat, uc_feat, on=['User_id', 'Coupon_id'], how='left')
    feat = pd.merge(feat, o_feat, on=['User_id', 'Coupon_id','Date_received'], how='left')

    # 不同特征块的交互特征
    feat['history_field_User_id_Merchant_id_receive_not_consume_rate_in_User_id'] = list(
        map(lambda x, y: x / y if
        y != 0 else 0, feat['history_field_User_id_Merchant_id_receive_not_consume_cnt'],
            feat['history_field_User_id_Merchant_id_receive_not_consume_cnt']))

    feat['history_field_User_id_Merchant_id_receive_and_consume_rate_in_User_id'] = list(
        map(lambda x, y: x / y if
        y != 0 else 0, feat['history_field_User_id_Merchant_id_receive_and_consume_cnt'],
            feat['history_field_User_id_Merchant_id_receive_and_consume_cnt']))

    feat['history_field_User_id_Merchant_id_receive_and_consume_rate_in_Merchant_id'] = list(
        map(lambda x, y: x / y if
        y != 0 else 0, feat['history_field_User_id_Merchant_id_receive_and_consume_cnt'],
            feat['history_field_User_id_Merchant_id_receive_and_consume_cnt']))

    return feat


'''middle'''
def middle_user_feature(label_field, history_field):
    # 源数据
    data = history_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(lambda x: int(x) if x == x else 0)
    data['Date_received'] = data['Date_received'].map(lambda x: int(x) if x == x else 0)
    data['cnt'] = 1

    # 主键
    keys = ['User_id']
    prefixs = 'middle_field_' + '_'.join(keys) + '_'
    u_feat = label_field[keys].drop_duplicates(keep='first')

    # 用户领券数
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户未核销数
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) == 'nan')], index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_not_consume_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户核销数
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'nan')], index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_and_consume_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户核销率
    u_feat[prefixs + 'receive_and_consume_rate'] = list(map(lambda x, y: x / y if y != 0 else 0,
                                                            u_feat[prefixs + 'receive_and_consume_cnt'],
                                                            u_feat[prefixs + 'receive_cnt']))
    # 用户领取了多少个不同商家的优惠券
    pivot = pd.pivot_table(data, index=keys, values='Merchant_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'Merchant_id': prefixs + 'receive_differ_Merchant_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户对领券商家的15天内的核销数
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='Merchant_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(
        columns={'Merchant_id': prefixs + 'receive_and_consume_differ_Merchant_cnt_15'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户对领券商家的15天内的核销率
    u_feat[prefixs + 'receive_and_consume_rate_15_differ_Merchant'] = list(map(lambda x, y: x / y if y != 0 else 0,
                                                                               u_feat[
                                                                                   prefixs + 'receive_and_consume_differ_Merchant_cnt_15'],
                                                                               u_feat[
                                                                                   prefixs + 'receive_differ_Merchant_cnt']))

    # 用户核销数/用户对领券商家15天内的核销数
    u_feat[prefixs + 'receive_and_consume_rate_15_differ_Merchant'] = list(map(lambda x, y: x / y if y != 0 else 0,
                                                                               u_feat[
                                                                                   prefixs + 'receive_and_consume_cnt'],
                                                                               u_feat[
                                                                                   prefixs + 'receive_and_consume_differ_Merchant_cnt_15']))

    # 用户15天内核销的最大距离
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='Distance',
                           aggfunc=lambda x: np.max([np.nan if i == -1 else i for i in x]))
    pivot = pd.DataFrame(pivot).rename(
        columns={'Distance': prefixs + 'receive_and_consume_15_max_Distance'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(-1, downcast='infer', inplace=True)

    # 用户15天内核销的最小距离
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='Distance',
                           aggfunc=lambda x: np.min([np.nan if i == -1 else i for i in x]))
    pivot = pd.DataFrame(pivot).rename(
        columns={'Distance': prefixs + 'receive_and_consume_15_min_Distance'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(-1, downcast='infer', inplace=True)

    # 用户15天内核销的平均距离
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='Distance',
                           aggfunc=lambda x: np.mean([np.nan if i == -1 else i for i in x]))
    pivot = pd.DataFrame(pivot).rename(
        columns={'Distance': prefixs + 'receive_and_consume_15_mean_Distance'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(-1, downcast='infer', inplace=True)

    # 用户领取的不同券的数量
    pivot = pd.pivot_table(data, index=keys, values='Coupon_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'Coupon_id': prefixs + 'receive_differ_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户15天内核销的不同券的数量
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='Coupon_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(
        columns={'Coupon_id': prefixs + 'receive_and_consume_15_differ_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户对不同券的15天内的核销率
    u_feat[prefixs + 'receive_and_consume_differ_rate_15'] = list(map(lambda x, y: x / y if y != 0 else 0,
                                                                      u_feat[
                                                                          prefixs + 'receive_and_consume_15_differ_cnt'],
                                                                      u_feat[prefixs + 'receive_differ_cnt']))

    # 用户15天内被核销的平均时间间隔
    tmp = data[data['label'] == 1]
    tmp['gap'] = (tmp['date'] - tmp['date_received']).map(lambda x: x.total_seconds() / (60 * 60 * 24))
    pivot = pd.pivot_table(tmp, index=keys, values='gap', aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={'gap': prefixs + 'consumed_mean_time_gap_15'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(-1, downcast='infer', inplace=True)

    # 用户15天内被核销的最小时间间隔
    pivot = pd.pivot_table(tmp, index=keys, values='gap', aggfunc=np.min)
    pivot = pd.DataFrame(pivot).rename(columns={'gap': prefixs + 'consumed_min_time_gap_15'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(-1, downcast='infer', inplace=True)

    # 用户15天内被核销的最小折扣率
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='discount_rate', aggfunc=np.min)
    pivot = pd.DataFrame(pivot).rename(
        columns={'discount_rate': prefixs + 'consumed_15_min_discount_rate'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户15天内被核销的最大折扣率
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='discount_rate', aggfunc=np.max)
    pivot = pd.DataFrame(pivot).rename(
        columns={'discount_rate': prefixs + 'consumed_15_max_discount_rate'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户15天内被核销的折扣率中位数
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='discount_rate', aggfunc=np.median)
    pivot = pd.DataFrame(pivot).rename(
        columns={'discount_rate': prefixs + 'consumed_15_median_discount_rate'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    return u_feat


def middle_merchant_feature(label_field, history_field):
    # 源数
    data = history_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(lambda x: int(x) if x == x else 0)
    data['Date_received'] = data['Date_received'].map(lambda x: int(x) if x == x else 0)
    data['cnt'] = 1

    # 主键
    keys = ['Merchant_id']
    prefixs = 'middle_field_' + '_'.join(keys) + '_'
    m_feat = label_field[keys].drop_duplicates(keep='first')

    # 商家的优惠券被领取次数
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)

    # 商家被不同客户领取次数：
    pivot = pd.pivot_table(data, index=keys, values='User_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'User_id': prefixs + 'received_differ_User_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)

    # 商家的券被核销的次数：
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'nan')], index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_and_consumed_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)

    # 商家的券的被核销率：
    m_feat[prefixs + 'received_and_consumed_rate'] = list(map(lambda x, y: x / y if y != 0 else 0,
                                                              m_feat[prefixs + 'received_and_consumed_cnt'],
                                                              m_feat[prefixs + 'received_cnt']))

    # 商家的券没被核销的次数：
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) == 'nan')], index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_not_consumed_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)

    # 商家提供的不同的优惠券数：
    pivot = pd.pivot_table(data, index=keys, values='Coupon_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'Coupon_id': prefixs + 'differ_Coupon_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)

    # 平均每个用户核销多少张商家的券
    m_feat[prefixs + 'per_User_consume_cnt'] = list(map(lambda x, y: x / y if y != 0 else 0,
                                                        m_feat[prefixs + 'received_and_consumed_cnt'],
                                                        m_feat[prefixs + 'received_differ_User_cnt']))

    # 商家提供的券平均被使用多少次
    m_feat[prefixs + 'per_Coupon_consume_cnt'] = list(map(lambda x, y: x / y if y != 0 else 0,
                                                          m_feat[prefixs + 'received_and_consumed_cnt'],
                                                          m_feat[prefixs + 'differ_Coupon_cnt']))

    # 商家15天内被核销的券种类数量
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='Coupon_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(
        columns={'Coupon_id': prefixs + 'received_and_consumed_15_differ_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)

    # 商家15天内核销的券距离用户的最大距离
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='Distance',
                           aggfunc=lambda x: np.max([np.nan if i == -1 else i for i in x]))
    pivot = pd.DataFrame(pivot).rename(
        columns={'Distance': prefixs + 'received_and_consumed_15_max_Distance'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(-1, downcast='infer', inplace=True)

    # 商家15天内核销的券距离用户的平均距离
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='Distance',
                           aggfunc=lambda x: np.mean([np.nan if i == -1 else i for i in x]))
    pivot = pd.DataFrame(pivot).rename(
        columns={'Distance': prefixs + 'received_and_consumed_15_mean_Distance'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(-1, downcast='infer', inplace=True)

    # 商家15天内核销的券距离用户的最小距离
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='Distance',
                           aggfunc=lambda x: np.min([np.nan if i == -1 else i for i in x]))
    pivot = pd.DataFrame(pivot).rename(
        columns={'Distance': prefixs + 'received_and_consumed_15_min_Distance'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(-1, downcast='infer', inplace=True)

    # 商家15天内核销的券的最大折扣率
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='discount_rate', aggfunc=np.max)
    pivot = pd.DataFrame(pivot).rename(
        columns={'discount_rate': prefixs + 'consumed_15_max_discount_rate'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)

    # 商家15天内核销的券的最小折扣率
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='discount_rate', aggfunc=np.min)
    pivot = pd.DataFrame(pivot).rename(
        columns={'discount_rate': prefixs + 'consumed_15_min_discount_rate'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)

    # 商家15天内核销的券的折扣率中位数
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='discount_rate', aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(
        columns={'discount_rate': prefixs + 'consumed_15_mean_discount_rate'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)

    # 商家15天内被核销的券的平均时间间隔
    tmp = data[data['label'] == 1]
    tmp['gap'] = (tmp['date'] - tmp['date_received']).map(lambda x: x.total_seconds() / (60 * 60 * 24))
    pivot = pd.pivot_table(tmp, index=keys, values='gap', aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={'gap': prefixs + 'consumed_mean_time_gap_15'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(-1, downcast='infer', inplace=True)

    # 商家15天内被核销的不同种类券占所有不同种类券的比例
    m_feat[prefixs + prefixs + 'received_and_consumed_differ_rate'] = list(map(lambda x, y: x / y if y != 0 else 0,
                                                        m_feat[prefixs + 'received_and_consumed_15_differ_cnt'],
                                                        m_feat[prefixs + 'differ_Coupon_cnt']))

    m_feat.fillna(0, downcast='infer', inplace=True)

    return m_feat


def middle_coupon_feature(label_field, history_field):
    # 源数
    data = history_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(lambda x: int(x) if x == x else 0)
    data['Date_received'] = data['Date_received'].map(lambda x: int(x) if x == x else 0)
    data['cnt'] = 1

    # 主键
    keys = ['Coupon_id']
    prefixs = 'middle_field_' + '_'.join(keys) + '_'
    c_feat = label_field[keys].drop_duplicates(keep='first')

    # 优惠券被领取次数
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_cnt'}).reset_index()
    c_feat = pd.merge(c_feat, pivot, on=keys, how='left')
    c_feat.fillna(0, downcast='infer', inplace=True)

    # 优惠券15天内被核销的次数：
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_and_consumed_cnt_15'}).reset_index()
    c_feat = pd.merge(c_feat, pivot, on=keys, how='left')
    c_feat.fillna(0, downcast='infer', inplace=True)

    # 优惠券15天内的核销率
    c_feat[prefixs + 'received_and_consumed_rate_15'] = list(map(lambda x, y: x / y if y != 0 else 0,
                                                                 c_feat[prefixs + 'received_and_consumed_cnt_15'],
                                                                 c_feat[prefixs + 'received_cnt']))

    # 优惠券15天内被核销的平均时间间隔
    tmp = data[data['label'] == 1]
    tmp['gap'] = (tmp['date'] - tmp['date_received']).map(lambda x: x.total_seconds() / (60 * 60 * 24))
    pivot = pd.pivot_table(tmp, index=keys, values='gap', aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={'gap': prefixs + 'consumed_mean_time_gap_15'}).reset_index()
    c_feat = pd.merge(c_feat, pivot, on=keys, how='left')
    c_feat.fillna(-1, downcast='infer', inplace=True)

    # 满减优惠券最低消费的中位数：
    pivot = pd.pivot_table(data[data['is_manjian'] == 1], index=keys, values='min_cost_of_manjian', aggfunc=np.median)
    pivot = pd.DataFrame(pivot).rename(
        columns={'min_cost_of_manjian': prefixs + 'median_of_min_cost_of_manjian'}).reset_index()
    c_feat = pd.merge(c_feat, pivot, on=keys, how='left')
    c_feat.fillna(0, downcast='infer', inplace=True)

    return c_feat


def middle_user_merchant_feature(label_field, history_field):
    # 源数据
    data = history_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(lambda x: int(x) if x == x else 0)
    data['Date_received'] = data['Date_received'].map(lambda x: int(x) if x == x else 0)
    data['cnt'] = 1

    # 主键
    keys = ['User_id', 'Merchant_id']
    prefixs = 'middle_field_' + '_'.join(keys) + '_'
    um_feat = label_field[keys].drop_duplicates(keep='first')

    # 该用户领取该商家的优惠券的数目
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    um_feat = pd.merge(um_feat, pivot, on=keys, how='left')
    um_feat.fillna(0, downcast='infer', inplace=True)

    # 该用户核销该商家的优惠券的数目
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'nan')], index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_and_consume_cnt'}).reset_index()
    um_feat = pd.merge(um_feat, pivot, on=keys, how='left')
    um_feat.fillna(0, downcast='infer', inplace=True)

    # 该用户核销该商家的优惠券的概率
    um_feat[prefixs + 'receive_and_consume_rate'] = list(map(lambda x, y: x / y if y != 0 else 0,
                                                                 um_feat[prefixs + 'receive_and_consume_cnt'],
                                                                 um_feat[prefixs + 'receive_cnt']))
    um_feat.fillna(0, downcast='infer', inplace=True)

    return um_feat


def middle_user_coupon_feature(label_field, history_field):
    # 源数据
    data = history_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(lambda x: int(x) if x == x else 0)
    data['Date_received'] = data['Date_received'].map(lambda x: int(x) if x == x else 0)
    data['cnt'] = 1

    # 主键
    keys = ['User_id', 'Coupon_id']
    prefixs = 'middle_field_' + '_'.join(keys) + '_'
    uc_feat = label_field[keys].drop_duplicates(keep='first')

    # 该用户领取该商家的优惠券数目
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    uc_feat = pd.merge(uc_feat, pivot, on=keys, how='left')
    uc_feat.fillna(0, downcast='infer', inplace=True)

    # 该用户核销该商家的优惠券数目
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'nan')], index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_and_consume_cnt'}).reset_index()
    uc_feat = pd.merge(uc_feat, pivot, on=keys, how='left')
    uc_feat.fillna(0, downcast='infer', inplace=True)

    # 该用户核销该商家的优惠券的概率
    uc_feat[prefixs + 'receive_and_consume_rate'] = list(map(lambda x, y: x / y if y != 0 else 0,
                                                             uc_feat[prefixs + 'receive_and_consume_cnt'],
                                                             uc_feat[prefixs + 'receive_cnt']))
    uc_feat.fillna(0, downcast='infer', inplace=True)

    return uc_feat

def middle_other_feature(label_field,history_field):
    data = history_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1

    keys = ['User_id', 'Merchant_id']
    feat = label_field[keys].drop_duplicates(keep='first')

    t = data[['User_id', 'Merchant_id', 'Date']].copy()
    t = t[t['Date'] != 'nan'][['User_id', 'Merchant_id']]
    t['user_merchant_buy_total'] = 1
    t = t.groupby(['User_id', 'Merchant_id']).agg('sum').reset_index()
    # 去重没有任何意义，因为agg(sum)的时候已经相当于去重了，reset_index之后也不会变回agg前的行数
    t.drop_duplicates(inplace=True)

    # 一个客户在一个商家浏览的次数（领过优惠券或者买过商品）
    t1 = data[['User_id', 'Merchant_id']]
    t1['user_merchant_any'] = 1
    t1 = t1.groupby(['User_id', 'Merchant_id']).agg('sum').reset_index()
    t1.drop_duplicates(inplace=True)

    keys1 = ['User_id']
    feat1 = label_field[keys1].drop_duplicates(keep='first')
    # 客户使用优惠券购买的次数
    t2 = data[(data['Date'] != 'nan') & (data['Coupon_id'] != 'nan')][['User_id']]
    t2['buy_use_coupon'] = 1
    t2 = t2.groupby('User_id').agg('sum').reset_index()

    # 客户购买任意商品的总次数
    t3 = data[data['Date'] != 'nan'][['User_id']]
    t3['buy_total'] = 1
    t3 = t3.groupby('User_id').agg('sum').reset_index()

    # 客户收到优惠券的总数
    t4 = data[data['Coupon_id'] != 'nan'][['User_id']]
    t4['coupon_received'] = 1
    t4 = t4.groupby('User_id').agg('sum').reset_index()

    keys2 = ['User_id','Merchant_id']
    feat2 = label_field[keys2].drop_duplicates(keep='first')
    # 卖出的商品
    t5 = data[data['Date'] != 'nan'][['Merchant_id']].copy()
    t5['total_sales'] = 1
    # 每个商品的销售数量
    t5 = t5.groupby('Merchant_id').agg('sum').reset_index()

    # 使用了优惠券消费的商品，正样本
    t6 = data[(data['Date'] != 'nan') & (
            data['Coupon_id'] != 'nan')][['Merchant_id']].copy()
    t6['sales_use_coupon'] = 1
    t6 = t6.groupby('Merchant_id').agg('sum').reset_index()

    t7 = data[['User_id', 'Coupon_id']].copy()
    t7['this_month_user_receive_same_coupn_count'] = 1
    t7 = t7.groupby(['User_id', 'Coupon_id']).agg('sum').reset_index()

    # t8:用户领取特定优惠券的最大时间和最小时间
    t8 = data[['User_id', 'Coupon_id', 'Date_received']].copy()
    t8['Date_received'] = t8['Date_received'].astype('str')
    t8 = t8.groupby(['User_id', 'Coupon_id'])['Date_received'].agg(
        lambda x: ':'.join(x)).reset_index()
    t8['receive_number'] = t8['Date_received'].apply(lambda s: len(s.split(':')))
    t8 = t8[t8['receive_number'] > 1]
    # 最大接受的日期
    t8['max_date_received'] = t8['Date_received'].apply(
        lambda s: max([int(d) for d in s.split(':')]))
    # 最小的接收日期
    t8['min_date_received'] = t8['Date_received'].apply(
        lambda s: min([int(d) for d in s.split(':')]))
    t8 = t8[['User_id', 'Coupon_id', 'max_date_received', 'min_date_received']]

    feat = pd.merge(feat, t, on=[
        'User_id', 'Merchant_id'], how='left')
    feat = pd.merge(feat, t1, on=[
        'User_id', 'Merchant_id'], how='left')
    feat1 = pd.merge(feat1, t2, on=[
        'User_id'], how='left')
    feat1 = pd.merge(feat1, t3, on=[
        'User_id'], how='left')
    feat1 = pd.merge(feat1, t4, on=[
        'User_id'], how='left')
    feat2 = pd.merge(
        feat2, t5, on='Merchant_id', how='left')
    feat2 = pd.merge(
        feat2, t6, on='Merchant_id', how='left')

    feat3 = data[['User_id', 'Coupon_id', 'Date_received']]
    feat3 = pd.merge(feat3, t7, on=['User_id', 'Coupon_id'], how='left')
    feat3 = pd.merge(feat3, t8, on=['User_id', 'Coupon_id'], how='left')
    feat3['this_month_user_receive_same_coupon_lastone'] = feat3['max_date_received'] - feat3['Date_received'].astype(int)
    feat3['this_month_user_receive_same_coupon_firstone'] = feat3['Date_received'].astype(
        int) - feat3['min_date_received']

    feat1['buy_use_coupon'] = feat1['buy_use_coupon'].replace(
        np.nan, 0)
    feat2['sales_use_coupon'] = feat2['sales_use_coupon'].replace(
        np.nan, 0)

    feat['user_merchant_rate'] = feat['user_merchant_buy_total'].astype(
        'float') / feat['user_merchant_any'].astype('float')
    feat1['buy_use_coupon_rate'] = feat1['buy_use_coupon'].astype('float') / feat1['buy_total'].astype(
        'float')
    feat1['user_coupon_transfer_rate'] = feat1['buy_use_coupon'].astype(
        'float') / feat1['coupon_received'].astype('float')
    feat2['coupon_rate'] = feat2['sales_use_coupon'].astype(
        'float') / feat2['total_sales']

    all_feat = pd.merge(feat1, feat, on=['User_id'], how='left')
    all_feat = pd.merge(all_feat, feat2, on=['User_id'], how='left')
    all_feat = pd.merge(all_feat, feat3, on=['User_id'], how='left')

    return all_feat


def middle_feature(lf, hf):
    u_feat = middle_user_feature(lf, hf)
    m_feat = middle_merchant_feature(lf, hf)
    c_feat = middle_coupon_feature(lf, hf)
    um_feat = middle_user_merchant_feature(lf, hf)
    uc_feat = middle_user_coupon_feature(lf, hf)
    o_feat = middle_other_feature(lf,hf)

    # 添加特征
    feat = lf.copy()
    feat = pd.merge(feat, u_feat, on=['User_id'], how='left')
    feat = pd.merge(feat, m_feat, on=['Merchant_id'], how='left')
    feat = pd.merge(feat, c_feat, on=['Coupon_id'], how='left')
    feat = pd.merge(feat, um_feat, on=['User_id', 'Merchant_id'], how='left')
    feat = pd.merge(feat, uc_feat, on=['User_id', 'Coupon_id'], how='left')
    feat = pd.merge(feat, o_feat, on=['User_id', 'Coupon_id','Date_received'], how='left')

    # 不同特征块的交互特征
    feat['middle_field_User_id_Merchant_id_receive_and_consume_rate_in_User_id'] = list(
        map(lambda x, y: x / y if
        y != 0 else 0, feat['middle_field_User_id_Merchant_id_receive_and_consume_cnt'],
            feat['middle_field_User_id_Merchant_id_receive_and_consume_cnt']))

    feat['middle_field_User_id_Merchant_id_receive_and_consume_rate_in_Merchant_id'] = list(
        map(lambda x, y: x / y if
        y != 0 else 0, feat['middle_field_User_id_Merchant_id_receive_and_consume_cnt'],
            feat['middle_field_User_id_Merchant_id_receive_and_consume_cnt']))

    return feat

'''label'''
def label_user_feature(label_field):
    # 源数据
    data = label_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1

    # 主键
    keys = ['User_id']
    prefixs = 'label_field_' + '_'.join(keys) + '_'
    u_feat = label_field[keys].drop_duplicates(keep='first')

    # 用户领券数
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户领取了多少个不同商家的优惠券
    pivot = pd.pivot_table(data, index=keys, values='Merchant_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'Merchant_id': prefixs + 'receive_differ_Merchant_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户领取的不同券的数量
    pivot = pd.pivot_table(data, index=keys, values='Coupon_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'Coupon_id': prefixs + 'receive_differ_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 热启动特征
    tmp = data[keys + ['Date_received']].sort_values(['Date_received'], ascending=True)
    first = tmp.drop_duplicates(keys, keep='first')
    first[prefixs + 'is_first_receive'] = 1
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    last = tmp.drop_duplicates(keys, keep='last')
    last[prefixs + 'is_last_receive'] = 1
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户领取满减优惠券的数目
    pivot = pd.pivot_table(data[data['is_manjian'] == 1], index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_manjian_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户领取非满减优惠券的数目
    pivot = pd.pivot_table(data[data['is_manjian'] != 1], index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_no_manjian_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户领取满减优惠券的概率
    u_feat[prefixs + 'receive_manjian_rate'] = list(map(lambda x, y: x / y if y != 0 else 0,
                                                         u_feat[prefixs + 'receive_manjian_cnt'],
                                                         u_feat[prefixs + 'received_cnt']))
    # 用户领取非满减优惠券的概率
    u_feat[prefixs + 'receive_no_manjian_rate'] = list(map(lambda x, y: x / y if y != 0 else 0,
                                                            u_feat[prefixs + 'receive_no_manjian_cnt'],
                                                            u_feat[prefixs + 'received_cnt']))

    return u_feat


def label_merchant_feature(label_field):
    # 源数
    data = label_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1

    # 主键
    keys = ['Merchant_id']
    prefixs = 'label_field_' + '_'.join(keys) + '_'
    m_feat = label_field[keys].drop_duplicates(keep='first')

    # 商家的优惠券被领取次数
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)

    # 商家被不同客户领取次数：
    pivot = pd.pivot_table(data, index=keys, values='User_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'User_id': prefixs + 'received_differ_User_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)

    # 商家提供的不同的优惠券数：
    pivot = pd.pivot_table(data, index=keys, values='Coupon_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'Coupon_id': prefixs + 'differ_Coupon_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)

    return m_feat


def label_coupon_feature(label_field):
    # 源数
    data = label_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1

    # 主键
    keys = ['Coupon_id']
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    c_feat = label_field[keys].drop_duplicates(keep='first')

    # 优惠券被领取次数
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_cnt'}).reset_index()
    c_feat = pd.merge(c_feat, pivot, on=keys, how='left')
    c_feat.fillna(0, downcast='infer', inplace=True)

    # 该优惠券被多少不同的用户领取
    pivot = pd.pivot_table(data, index=keys, values='User_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'User_id': prefixs + 'receive_differ_User_cnt'}).reset_index()
    c_feat = pd.merge(c_feat, pivot, on=keys, how='left')
    c_feat.fillna(0, downcast='infer', inplace=True)

    # 该优惠券被多少不同的商家发放
    pivot = pd.pivot_table(data, index=keys, values='Merchant_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'Merchant': prefixs + 'receive_differ_Merchant_cnt'}).reset_index()
    c_feat = pd.merge(c_feat, pivot, on=keys, how='left')
    c_feat.fillna(0, downcast='infer', inplace=True)

    return c_feat


def label_user_merchant_feature(label_field):
    # 源数据
    data = label_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1

    # 主键
    keys = ['User_id', 'Merchant_id']
    prefixs = 'label_field_' + '_'.join(keys) + '_'
    um_feat = label_field[keys].drop_duplicates(keep='first')

    # 该用户领取该商家的优惠券的数目
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_cnt'}).reset_index()
    um_feat = pd.merge(um_feat, pivot, on=keys, how='left')
    um_feat.fillna(0, downcast='infer', inplace=True)

    tmp = data[keys + ['Date_received']].sort_values(['Date_received'], ascending=True)
    first = tmp.drop_duplicates(keys, keep='first')
    first[prefixs + 'is_first_receive'] = 1
    um_feat = pd.merge(um_feat, pivot, on=keys, how='left')
    um_feat.fillna(0, downcast='infer', inplace=True)

    last = tmp.drop_duplicates(keys, keep='last')
    last[prefixs + 'is_last_receive'] = 1
    um_feat = pd.merge(um_feat, pivot, on=keys, how='left')
    um_feat.fillna(0, downcast='infer', inplace=True)

    #该用户在该商家领取满减优惠券的数目
    pivot = pd.pivot_table(data[data['is_manjian'] == 1], index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_manjian_cnt'}).reset_index()
    um_feat = pd.merge(um_feat, pivot, on=keys, how='left')
    um_feat.fillna(0, downcast='infer', inplace=True)

    # 该用户在该商家领取非满减优惠券的数目
    pivot = pd.pivot_table(data[data['is_manjian'] != 1], index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_no_manjian_cnt'}).reset_index()
    um_feat = pd.merge(um_feat, pivot, on=keys, how='left')
    um_feat.fillna(0, downcast='infer', inplace=True)

    # 该用户在该商家领取满减优惠券的概率
    um_feat[prefixs + 'receive_manjian_rate'] = list(map(lambda x, y: x / y if y != 0 else 0,
                                                             um_feat[prefixs + 'receive_manjian_cnt'],
                                                             um_feat[prefixs + 'received_cnt']))
    # 该用户在该商家领取非满减优惠券的概率
    um_feat[prefixs + 'receive_no_manjian_rate'] = list(map(lambda x, y: x / y if y != 0 else 0,
                                                         um_feat[prefixs + 'receive_no_manjian_cnt'],
                                                         um_feat[prefixs + 'received_cnt']))

    return um_feat


def label_discount_feature(label_field):
    # 源数据
    data = label_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1

    # 主键
    keys = ['Discount_rate']
    prefixs = 'label_field_' + '_'.join(keys) + '_'
    d_feat = label_field[keys].drop_duplicates(keep='first')

    # 该用户领取该商家的优惠券的数目
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    d_feat = pd.merge(d_feat, pivot, on=keys, how='left')
    d_feat.fillna(0, downcast='infer', inplace=True)

    # 该折扣下有多少种不同的优惠券
    pivot = pd.pivot_table(data, index=keys, values='Coupon_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'Coupon_id': prefixs + 'receive_differ_Coupon_cnt'}).reset_index()
    d_feat = pd.merge(d_feat, pivot, on=keys, how='left')
    d_feat.fillna(0, downcast='infer', inplace=True)

    # 该折扣券有多少不同的商家发放
    pivot = pd.pivot_table(data, index=keys, values='Merchant_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'Merchant': prefixs + 'receive_differ_Merchant_cnt'}).reset_index()
    d_feat = pd.merge(d_feat, pivot, on=keys, how='left')
    d_feat.fillna(0, downcast='infer', inplace=True)

    return d_feat


def label_user_coupon_feature(label_field):
    # 源数据
    data = label_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1

    # 主键
    keys = ['User_id', 'Coupon_id']
    prefixs = 'label_field_' + '_'.join(keys) + '_'
    uc_feat = label_field[keys].drop_duplicates(keep='first')

    # 该用户领取该商家的优惠券数目
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    uc_feat = pd.merge(uc_feat, pivot, on=keys, how='left')
    uc_feat.fillna(0, downcast='infer', inplace=True)

    tmp = data[keys + ['Date_received']].sort_values(['Date_received'], ascending=True)
    first = tmp.drop_duplicates(keys, keep='first')
    first[prefixs + 'is_first_receive'] = 1
    uc_feat = pd.merge(uc_feat, pivot, on=keys, how='left')
    uc_feat.fillna(0, downcast='infer', inplace=True)

    last = tmp.drop_duplicates(keys, keep='last')
    last[prefixs + 'is_last_receive'] = 1
    uc_feat = pd.merge(uc_feat, pivot, on=keys, how='left')
    uc_feat.fillna(0, downcast='infer', inplace=True)

    return uc_feat


def label_user_discount_feature(label_field):
    # 源数据
    data = label_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1

    # 主键
    keys = ['User_id', 'discount_rate']
    prefixs = 'label_field_' + '_'.join(keys) + '_'
    ud_feat = label_field[keys].drop_duplicates(keep='first')

    # 该用户领取该折扣率优惠券数目
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    ud_feat = pd.merge(ud_feat, pivot, on=keys, how='left')
    ud_feat.fillna(0, downcast='infer', inplace=True)

    # 该用户领取不同折扣率的优惠券数目
    pivot = pd.pivot_table(data, index=keys, values='Coupon_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'Coupon_id': prefixs + 'receive_differ_cnt'}).reset_index()
    ud_feat = pd.merge(ud_feat, pivot, on=keys, how='left')
    ud_feat.fillna(0, downcast='infer', inplace=True)

    # 该用户核商家的不同折扣率优惠券的数目
    pivot = pd.pivot_table(data, index=keys, values='Merchant_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'Merchant_id': prefixs + 'receive_differ_merchant_cnt'}).reset_index()
    ud_feat = pd.merge(ud_feat, pivot, on=keys, how='left')
    ud_feat.fillna(0, downcast='infer', inplace=True)

    return ud_feat


def label_feature(lf):
    u_feat = label_user_feature(lf)
    m_feat = label_merchant_feature(lf)
    c_feat = label_coupon_feature(lf)
    d_feat = label_discount_feature(lf)
    um_feat = label_user_merchant_feature(lf)
    uc_feat = label_user_coupon_feature(lf)
    ud_feat = label_user_discount_feature(lf)

    # 添加特征
    share = list(set(u_feat.columns.tolist()) & set(m_feat.columns.tolist()))
    feat = pd.concat([u_feat, m_feat.drop(share, axis=1)], axis=1)
    feat = pd.concat([feat, c_feat.drop(share, axis=1)], axis=1)
    feat = pd.concat([feat, d_feat.drop(share, axis=1)], axis=1)
    feat = pd.concat([feat, um_feat.drop(share, axis=1)], axis=1)
    feat = pd.concat([feat, uc_feat.drop(share, axis=1)], axis=1)
    feat = pd.concat([feat, ud_feat.drop(share, axis=1)], axis=1)

    return feat
