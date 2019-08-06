# label_field_feature.py

import pandas as pd
import numpy as np


def get_label_field_user_feature(label_field):
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
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
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


def get_label_field_merchant_feature(label_field):
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


def get_label_field_coupon_feature(label_field):
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


def get_label_field_user_merchant_feature(label_field):
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
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
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


def get_label_field_discount_feature(label_field):
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


def get_label_field_user_coupon_feature(label_field):
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


def get_label_field_user_discount_feature(label_field):
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


def get_label_field_feature(label_field):
    u_feat = get_label_field_user_feature(label_field)
    m_feat = get_label_field_merchant_feature(label_field)
    c_feat = get_label_field_coupon_feature(label_field)
    d_feat = get_label_field_discount_feature(label_field)
    um_feat = get_label_field_user_merchant_feature(label_field)
    uc_feat = get_label_field_user_coupon_feature(label_field)
    ud_feat = get_label_field_user_discount_feature(label_field)

    # 添加特征
    share_characters = list(set(u_feat.columns.tolist()) & set(m_feat.columns.tolist()))
    label_feat = pd.concat([u_feat, m_feat.drop(share_characters, axis=1)], axis=1)
    label_feat = pd.concat([label_feat, c_feat.drop(share_characters, axis=1)], axis=1)
    label_feat = pd.concat([label_feat, d_feat.drop(share_characters, axis=1)], axis=1)
    label_feat = pd.concat([label_feat, um_feat.drop(share_characters, axis=1)], axis=1)
    label_feat = pd.concat([label_feat, uc_feat.drop(share_characters, axis=1)], axis=1)
    label_feat = pd.concat([label_feat, ud_feat.drop(share_characters, axis=1)], axis=1)

    label_feat.drop(['cnt'], axis=1, inplace=True)

    return label_feat
