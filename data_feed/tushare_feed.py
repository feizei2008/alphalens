import time
from typing import List
import tushare as ts
import pandas as pd
from pandas import DataFrame, Series
import os
import csv
import random
from datetime import datetime, timedelta
from utils.config_reader import ConfigReader
from utils.log import logger
from utils.common import get_file_list

config_reader = ConfigReader('config/api.ini')
token = config_reader.read_config('tushare', 'token')
pro = ts.pro_api(token)


def get_symbols_tushare():
    df_list = []
    t_d = datetime.today()
    print(t_d)
    exchanges = ['DCE', 'SHFE', 'CZCE', 'CFFEX', 'INE']
    fields = 'ts_code,symbol,exchange,name,fut_code,multiplier,trade_unit,per_unit,quote_unit,list_date,delist_date'
    for e in exchanges:
        df = pro.fut_basic(exchange=e, fut_type='1', fields=fields)
        df['delist_date'] = pd.to_datetime(df['delist_date'])
        # df = df.loc[df['delist_date'] > t_d]
        df_list.append(df)
    combined_df = pd.concat(df_list)
    return combined_df


def get_dominant_name_tushare(df: DataFrame, path, start_date: str, end_date: str):
    name_list = []
    # https://tushare.pro/document/2?doc_id=134
    e_d = {'CZCE': 'ZCE', 'SHFE': 'SHF', 'DCE': 'DCE', 'CFFEX': 'CFX', 'INE': 'INE'}
    df['exchange'] = df['exchange'].map(e_d)
    for e in df['exchange'].drop_duplicates():
        if os.path.exists(os.path.join(path, e)):
            pass
        else:
            os.makedirs(os.path.join(path, e))
        temp_df = df.loc[df.exchange == e]
        for f in temp_df['fut_code'].drop_duplicates():
            dominant_symbol = f'{f}.{e}'
            name = pro.fut_mapping(ts_code=dominant_symbol, start_date=start_date, end_date=end_date)
            name = name.sort_values(by='trade_date', ascending=True)
            name = name.set_index(name['trade_date'])
            name.drop(columns=['trade_date'], inplace=True)
            name['mapping_ts_code'] = tushare_domin_adj(name['mapping_ts_code'])
            name_list.append(name)
    names = pd.concat(name_list)
    return names


def get_dominant_k_lines_tushare(df: DataFrame, path, trade_cal: DataFrame):
    cnt = 0
    df['exchange'] = df['ts_code'].map(lambda x: x.split('.')[1])
    all_cnt = df['ts_code'].drop_duplicates().size
    for c in df['ts_code'].drop_duplicates():
        temp_df = df.loc[df.ts_code == c]
        # tushare date may not in correct order
        # temp_df['trade_date'] = pd.to_datetime(temp_df['trade_date'], format='%Y%m%d')
        temp_df = temp_df.sort_values(by='trade_date', ascending=True)
        begin, end = temp_df['trade_date'].iloc[0], temp_df['trade_date'].iloc[-1]
        full_cals = trade_cal.loc[(trade_cal.cal_date >= begin) & (trade_cal.cal_date <= end)]
        ex = temp_df['exchange'].iloc[0]
        c_list = []
        cal_cnt = 0
        mapping_dic = {}
        for m in temp_df['mapping_ts_code'].drop_duplicates():
            temp_df1 = temp_df.loc[temp_df.mapping_ts_code == m]
            begin1, end1 = temp_df1['trade_date'].iloc[0], temp_df1['trade_date'].iloc[-1]
            mapping_dic[m] = begin1, end1
            cal1 = trade_cal.loc[(trade_cal.cal_date >= begin1) & (trade_cal.cal_date <= end1)]
            cal_cnt += cal1.shape[0]
        if cal_cnt == full_cals.shape[0]:
            logger.info(f'{c}, cals:{full_cals.shape[0]}, k_line:{cal_cnt}, '
                        f'data correct: {full_cals.shape[0]==cal_cnt}, begin download.')
            for k, v in mapping_dic.items():
                time.sleep(random.randint(3, 8))
                k_lines = pro.fut_daily(ts_code=k, start_date=str(v[0]), end_date=str(v[1]))
                c_list.append(k_lines)
                logger.info(f'{k} k_lines {v[0]} to {v[1]} downloaded')
            c_domain_k = pd.concat(c_list)
            # c_domain_k['trade_date'] = pd.to_datetime(c_domain_k['trade_date'], format='%Y%m%d')
            c_domain_k = c_domain_k.sort_values(by='trade_date', ascending=True)
            logger.info(f'{c}, combined domain len:{c_domain_k.shape[0]}, trade cal len:{full_cals.shape[0]}, '
                        f'check: {c_domain_k.shape[0] == full_cals.shape[0]}')
            c_domain_k.to_csv(os.path.join(path, ex, f'{c}_{begin}-{end}_daily.csv'))
        elif cal_cnt != full_cals.shape[0]:
            logger.info(f'{c}, cals:{full_cals.shape[0]}, k_line:{cal_cnt}, '
                        f'data correct: {full_cals.shape[0]==cal_cnt}, do not download.')
        cnt += 1
        print(f'{cnt} of {all_cnt} completed')


# 获取交易日历
def get_cal_date(start: str, end: str) -> DataFrame:
    cal_date = pro.trade_cal(exchange='', start_date=start, end_date=end)
    cal_date = cal_date[cal_date.is_open == 1]
    cal_date = cal_date.sort_values(by='cal_date', ascending=True)
    # dates = cal_date.cal_date.values
    return cal_date


def download_tushare(data_path, tushare_path, start: str, end: str):
    # 获取各个期货交易所的全部合约
    if not os.path.exists(os.path.join(data_path, 'avail_fut_symbols.csv')):
        symbols = get_symbols_tushare()
        symbols.to_csv(data_path + 'avail_fut_symbols.csv')
    else:
        symbols = pd.read_csv(data_path + 'avail_fut_symbols.csv')
    # 获取各个品种在交易日历的主力合约代码
    if not os.path.exists(os.path.join(data_path, 'domain_names.csv')):
        domain_names = get_dominant_name_tushare(df=symbols, path=tushare_path, start_date=start, end_date=end)
        domain_names.to_csv(data_path + 'domain_names.csv')
    else:
        domain_names = pd.read_csv(data_path + 'domain_names.csv')
    # get trade cal data
    if not os.path.exists(os.path.join(data_path, 'trade_cal.csv')):
        trade_cal = get_cal_date(start, end)
        trade_cal.to_csv(data_path + 'trade_cal.csv')
    else:
        trade_cal = pd.read_csv(data_path + 'trade_cal.csv')
    # 获取主力合约的日线数据
    get_dominant_k_lines_tushare(df=domain_names, path=tushare_path, trade_cal=trade_cal)
    return


def tushare_domin_adj(Ser: Series):
    """
    tushare fut_mapping bug, below is an example:
    20180509-fu1806, 20180510-fu1905, 20180511-fu1806
    but this func still cannot fully solve the problem
    :param Ser:
    :return:
    """
    symbol = Ser.map(lambda x: x.split('.')[0])
    # exchange = Ser.map(lambda x: x.split('.')[1])
    year_month = symbol.map(lambda x: ''.join(list(filter(str.isdigit, x)))).astype(int)
    diff = year_month - year_month.shift(1)
    diff.fillna(0, inplace=True)
    back_forth = diff[diff != 0] + diff[diff != 0].shift(-1)
    wrong_domain = back_forth[back_forth == 0]
    if wrong_domain.size > 0:
        for k, v in wrong_domain.items():
            before_wrong_idx = Ser.index[Ser.index.get_loc(k) - 1]
            after_wrong_idx = Ser.index[Ser.index.get_loc(k) + 1]
            # print(before_wrong_idx, k, after_wrong_idx)
            print(f'wrong domain: \n'
                  f'{Ser.loc[before_wrong_idx:after_wrong_idx]}')
            Ser.loc[k] = Ser.loc[before_wrong_idx]
            print(f'domain has been corrected as: \n'
                  f'{Ser.loc[before_wrong_idx:after_wrong_idx]}')
    return Ser


def continuous_contract_adj(path):
    """
    the earliest contract is the base data and adjust the later ones
    :param path:
    :return:
    """
    files = get_file_list(path, [])
    for f in files:
        print(f)
        df = pd.read_csv(f)
        try:
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        except:
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y-%m-%d')
        df = df.sort_values(by='trade_date', ascending=True)
        df = df.set_index(df['trade_date'])
        try:
            df.drop(columns=['Unnamed: 0'], inplace=True)
            df.drop(columns=['trade_date'], inplace=True)
        except:
            pass
        symbol_ex = df['ts_code']
        symbol = symbol_ex.map(lambda x: x.split('.')[0])
        # exchange = Ser.map(lambda x: x.split('.')[1])
        year_month = symbol.map(lambda x: ''.join(list(filter(str.isdigit, x)))).astype(int)
        diff = year_month - year_month.shift(1)
        adj_mark = diff[diff != 0][1:]
        print(df.loc[adj_mark.index])

        if adj_mark.shape[0] != 0:
            for k, v in adj_mark.items():
                pre_idx = df.index[df.index.get_loc(k) - 1]
                df.loc[k, 'pre_date_old2new'] = df.loc[pre_idx, 'close'] / df.loc[k, 'pre_close']
            df['cont_adj'] = df['pre_date_old2new'].cumprod()
            df['cont_adj'].fillna(method='ffill', inplace=True)
            df['cont_adj'].fillna(1, inplace=True)
            df['close_adj'] = df['close'] * df['cont_adj']
        elif adj_mark.shape[0] == 0:
            df['cont_adj'] = 1
            df['close_adj'] = df['close'] * df['cont_adj']
        df.to_csv(f)


if __name__ == '__main__':
    """下载并处理日线数据，文件夹位置是老的，暂时不动"""
    t0 = datetime.now().strftime('%Y%m%d')
    start, end = '20180101', '20210618'
    father_dir = os.path.abspath(os.path.dirname(os.getcwd()))  # 上一级目录
    grandfather_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # 上上级目录
    data_path = os.path.join(father_dir, "data\\")
    tushare_path = os.path.join(father_dir, "data", "tushare\\")

    """下载起始日期内全部多品种合约，合并成主力合约"""
    download_tushare(data_path=data_path, tushare_path=tushare_path, start=start, end=end)
    """换月因子生成"""
    continuous_contract_adj(path=tushare_path)

    # domain_names = pd.read_csv(data_path + 'domain_names.csv')
    # fu = domain_names.loc[domain_names.ts_code == 'BB.DCE', 'mapping_ts_code']
    # tushare_domin_adj(fu)

