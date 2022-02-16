from pyecharts import Bar
import matplotlib
from talib import MOM
import os
import pandas as pd
import alphalens as al
from pylab import mpl
import matplotlib.pyplot as plt

# %matplotlib inline

# 正常显示画图时出现的中文和负号
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

print(al.__version__)

father_dir = os.path.abspath(os.path.dirname(os.getcwd()))  # 上一级目录
grandfather_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # 上上级目录，snowballArb根目录
prices_path = os.path.join(father_dir, 'data', 'processed', 'prices\\')
factors_path = os.path.join(father_dir, 'data', 'processed', 'factors\\')

if __name__ == '__main__':
    prices = pd.read_csv(os.path.join(prices_path, 'min_prices.csv'), parse_dates=True, index_col=['date'])
    factors = pd.read_csv(os.path.join(factors_path, 'min_factors.csv'), parse_dates=True, index_col=['date', 'asset'])
    # pandas大于1版本报错：RuntimeError: Cannot set name on a level of a MultiIndex. Use 'MultiIndex.set_names' instead.
    # 解决办法：https://github.com/quantopian/alphalens/pull/364/files
    """
    因子值分组规则：因子值小的排在第一组，因子值大的排在最后一组
    https://github.com/quantopian/alphalens/blob/master/alphalens/examples/alphalens_tutorial_on_quantopian.ipynb
    quantiles option chooses the buckets to have the same number of items but it doesn't take into
    consideration the factor values. For this reason there is another option bins, which chooses the
    buckets to be evenly spaced according to the values themselves.
    """

    forward_returns = al.utils.compute_forward_returns(factor=factors,
                                                       prices=prices,
                                                       periods=(1, 3, 5),
                                                       filter_zscore=20,
                                                       cumulative_returns=True)

    # factor_data = al.utils.get_clean_factor_and_forward_returns(factor=factors,
    #                                                             prices=prices,
    #                                                             groupby=None,
    #                                                             binning_by_group=False,
    #                                                             quantiles=5,  # None
    #                                                             bins=None,  # [0, 0.2, 0.4, 0.8, 1.0]
    #                                                             periods=(1, 3, 5),
    #                                                             filter_zscore=20,
    #                                                             groupby_labels=None,
    #                                                             max_loss=0.35,
    #                                                             zero_aware=False,
    #                                                             cumulative_returns=True)
    # al.tears.create_full_tear_sheet(factor_data, long_short=True, group_neutral=False, by_group=False)
    # al.tears.create_event_returns_tear_sheet(factor_data, prices,
    #                                          avgretplot=(5, 15),
    #                                          long_short=True,
    #                                          group_neutral=False,
    #                                          std_bar=True,
    #                                          by_group=False)
    #
    # al.tears.create_summary_tear_sheet(factor_data, long_short=True, group_neutral=False)
    # al.tears.create_returns_tear_sheet(factor_data, long_short=True, group_neutral=False, by_group=False)
    # al.tears.create_information_tear_sheet(factor_data, group_neutral=False, by_group=False)
    # al.tears.create_turnover_tear_sheet(factor_data)
    # ic = al.performance.mean_information_coefficient(factor_data, by_time='1y')
    #
    # attr = ic.index.strftime('%Y')
    # v1 = list(ic['1D'].round(2))
    # v2 = list(ic['3D'].round(2))
    # v3 = list(ic['5D'].round(2))
    # bar = Bar('IC均值：2006-2019')
    # bar.add('1D', attr, v1)
    # bar.add('3D', attr, v2)
    # bar.add('5D', attr, v3)
    # bar.render()
    #
    #

