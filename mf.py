# middle_field_feature.py

import pandas as pd
import numpy as np


def get_history_field_user_feature(label_field, history_field):
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


def get_history_field_merchant_feature(label_field, history_field):
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


def get_history_field_coupon_feature(label_field, history_field):
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


def get_history_field_user_merchant_feature(label_field, history_field):
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


def get_history_field_user_coupon_feature(label_field, history_field):
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


def get_middle_field_feature(label_feild, history_field):
    u_feat = get_history_field_user_feature(label_feild, history_field)
    m_feat = get_history_field_merchant_feature(label_feild, history_field)
    c_feat = get_history_field_coupon_feature(label_feild, history_field)
    um_feat = get_history_field_user_merchant_feature(label_feild, history_field)
    uc_feat = get_history_field_user_coupon_feature(label_feild, history_field)

    # 添加特征
    history_feat = label_feild.copy()
    history_feat = pd.merge(history_feat, u_feat, on=['User_id'], how='left')
    history_feat = pd.merge(history_feat, m_feat, on=['Merchant_id'], how='left')
    history_feat = pd.merge(history_feat, c_feat, on=['Coupon_id'], how='left')
    history_feat = pd.merge(history_feat, um_feat, on=['User_id', 'Merchant_id'], how='left')
    history_feat = pd.merge(history_feat, uc_feat, on=['User_id', 'Coupon_id'], how='left')

    # 不同特征块的交互特征
    history_feat['middle_field_User_id_Merchant_id_receive_and_consume_rate_in_User_id'] = list(
        map(lambda x, y: x / y if
        y != 0 else 0, history_feat['middle_field_User_id_Merchant_id_receive_and_consume_cnt'],
            history_feat['middle_field_User_id_Merchant_id_receive_and_consume_cnt']))

    history_feat['middle_field_User_id_Merchant_id_receive_and_consume_rate_in_Merchant_id'] = list(
        map(lambda x, y: x / y if
        y != 0 else 0, history_feat['middle_field_User_id_Merchant_id_receive_and_consume_cnt'],
            history_feat['middle_field_User_id_Merchant_id_receive_and_consume_cnt']))

    return history_feat
