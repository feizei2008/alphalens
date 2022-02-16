import os
import pandas as pd
import numpy as np
import tushare as ts
from matplotlib import pyplot as plt
from talib import MOM
from utils.common import get_file_list
from data_feed.tushare_feed import get_cal_date


father_dir = os.path.abspath(os.path.dirname(os.getcwd()))  # 上一级目录
grandfather_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # 上上级目录

# data = pd.read_csv(os.path.join(data_path, 'SHF\\', 'FU.SHF_20180102-20210528_daily.csv'))
# data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
# data = data.sort_values(by='trade_date', ascending=True)
# data = data.set_index(data['trade_date'])
# data = data.drop(columns=['Unnamed: 0', 'trade_date'])
# data['mom'] = MOM(data.pre_close, 5)
# print(data.index)

# # 创建MultiIndex
# # https://blog.csdn.net/sinat_26811377/article/details/98469964
# date_idx = [val for val in factor_val.index for i in range(len(factor_val.columns))]
# asset_idx = factor_val.columns.to_list() * len(factor_val.index)
# # print(factor_val.columns)
# # print(asset_idx)
# # print(len(date_idx), len(asset_idx))
# # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.MultiIndex.html
# multi_idx = pd.MultiIndex.from_arrays([date_idx, asset_idx], names=('trade_date', 'asset'))
# print(multi_idx)

# # alphalens不接受日频datetimeIndex之外的index
# prices = pd.read_csv('prices.csv', parse_dates=True, index_col=['date'])
# zz = pd.read_csv('000905.SH.csv', parse_dates=True, index_col=['date_time'])
# lens = prices.shape[0]
# new_idx = zz.index[:lens]
# prices['date'] = new_idx
# prices.set_index('date', inplace=True)
# prices.to_csv('prices.csv')
# # print(prices.head())


def gen_al_prices(source: str, path_list, start: str, end: str, trade_cal: pd.DataFrame) -> pd.DataFrame():
    ser_l, files = [], []
    for path in path_list:
        files += get_file_list(path, [])
    if source == 'tushare':  # tushare下载的为日频数据
        for f in files:
            symbol = f.split('\\')[-1].split('.')[0]
            date_range = f.split('\\')[-1].split('.')[1].split('_')[1]
            b, e = date_range.split('-')[0], date_range.split('-')[1]
            if (b == start) & (e == end):
                temp = pd.read_csv(f)
                temp.rename(columns={'trade_date': 'date'}, inplace=True)
                temp['date'] = pd.to_datetime(temp['date'], format='%Y-%m-%d')
                temp = temp.sort_values(by='date', ascending=True)
                temp = temp.set_index(temp['date'])
                ser = temp['close_adj']
                ser.rename(symbol, inplace=True)
                if ser.size == trade_cal.shape[0]:
                    ser_l.append(ser)
                    print(f'{symbol} appended.')
                elif ser.size != trade_cal.shape[0]:
                    print(f'{symbol} k_line len: {ser.size}, cal len: {trade_cal.shape[0]}, dropped.')
        df = pd.concat(ser_l, axis=1)
        return df
    elif source == 'tq':  # 天勤下载的为分钟频数据
        for f in files:
            symbol = f.split('\\')[-1].split('.')[1]
            temp = pd.read_csv(f, index_col='dt', parse_dates=True)
            ser = temp['close']
            ser.rename(symbol, inplace=True)
            ser_l.append(ser)
            print(f'{symbol} appended.')
        df = pd.concat(ser_l, axis=1)
        df.index.rename('date', inplace=True)
        df.dropna(inplace=True)
        return df


def pv_factor_sample(df: pd.DataFrame):
    # 生成因子值文件
    L = []
    for col in df.columns:
        # X = MOM(prices[col], 5)
        # 无负号动量因子，有负号反转因子
        X = np.log(prices[col] / prices[col].shift(5))
        X.name = col
        L.append(X)
    factors = pd.concat(L, axis=1)
    return factors


def gen_multi_index_factors(factors: pd.DataFrame):
    factors = factors.T
    factors['asset'] = factors.index
    factors = factors.T
    factors = factors[:-1]
    row_L = []
    for i in factors.index:
        row = factors.loc[i]
        row = pd.DataFrame(row)
        row['date'] = row.columns[0]
        row.rename(columns={row.columns[0]: 'factor_value'}, inplace=True)
        row['asset'] = row.index
        row = row.reindex(columns=['date', 'asset', 'factor_value'])
        row_L.append(row)
    factor_mi = pd.concat(row_L)
    factor_mi.fillna(0, inplace=True)

    # https://stackoverflow.com/questions/42236701/turn-columns-into-multi-level-index-pandas
    factor_mi = factor_mi.set_index(['date', 'asset'], drop=True)
    # print(factor_mi.index.levels)
    # factor_mi = pd.Series(factor_mi['factor_value'].values, index=factor_mi.index)

    return factor_mi


"""https://zhuanlan.zhihu.com/p/68613067"""


def winsor_data(data):
    q = data.quantile([0.02, 0.98])
    data[data < q.iloc[0]] = q.iloc[0]
    data[data > q.iloc[1]] = q.iloc[1]
    return data


# 数据标准化
def MaxMinNormal(data):
    """[0,1] normaliaztion"""
    x = (data - data.min()) / (data.max() - data.min())
    return x


if __name__ == '__main__':
    DCE_path = os.path.join(father_dir, 'data', 'daily_bar', 'tushare', 'DCE\\')
    SHF_path = os.path.join(father_dir, 'data', 'daily_bar', 'tushare', 'SHF\\')
    ZCE_path = os.path.join(father_dir, 'data', 'daily_bar', 'tushare', 'ZCE\\')
    tq_path = os.path.join(father_dir, 'data', 'min_bar', 'tq\\')

    prices_path = os.path.join(father_dir, 'data', 'processed', 'prices\\')
    factors_path = os.path.join(father_dir, 'data', 'processed', 'factors\\')
    start, end = '20180102', '20210601'
    trade_cal = get_cal_date(start, end)

    """日频价格和因子文件生成"""
    # # 保存价格文件
    # prices = gen_al_prices(source='tushare', path_list=[DCE_path, SHF_path, ZCE_path], start=start, end=end,
    #                        trade_cal=trade_cal)
    # prices.to_csv(os.path.join(prices_path, 'daily_prices.csv'))
    # # 用价格文件生成简单的价量因子做示范
    # factors = pv_factor_sample(prices)
    # # 生成双重索引['date', 'asset']格式的因子值文件
    # factors = gen_multi_index_factors(factors)
    # factors.to_csv(os.path.join(factors_path, 'daily_factors.csv'))
    #
    # # 因子归一化, CTA策略不要取极值
    # # factors_new = factors['factor_value'].groupby('date').apply(winsor_data)
    # factors_new = factors.groupby('date').apply(MaxMinNormal)
    # factors_new.hist(figsize=(12, 6), bins=20)
    # factors_new.to_csv(os.path.join(factors_path, 'daily_factors_new.csv'))
    # plt.show()

    """分钟频价格和因子文件生成"""
    # 保存价格文件
    prices = gen_al_prices(source='tq', path_list=[tq_path], start=start, end=end, trade_cal=trade_cal)
    prices.to_csv(os.path.join(prices_path, 'min_prices.csv'))
    # 用价格文件生成简单的价量因子做示范
    factors = pv_factor_sample(prices)
    # 生成双重索引['date', 'asset']格式的因子值文件
    factors = gen_multi_index_factors(factors)
    factors.to_csv(os.path.join(factors_path, 'min_factors.csv'))

    # 因子归一化, CTA策略不要取极值
    # factors_new = factors['factor_value'].groupby('date').apply(winsor_data)
    factors_new = factors.groupby('date').apply(MaxMinNormal)
    factors_new.hist(figsize=(12, 6), bins=20)
    factors_new.to_csv(os.path.join(factors_path, 'min_factors_new.csv'))
    plt.show()

    # res.dropna(inplace=True)
    # factor_cols = list(filter(lambda x: 'MOM' in x, res.columns.tolist()))
    # factor = res[factor_cols]
    # res = res.set_index([index_a, index_b])
