import sys
import os

import asyncio
import multiprocessing
from time import sleep
import threading
from threading import Timer
import math
import numpy as np
from tqsdk import (TqApi, TqSim, TqKq, TqAuth, TqAccount, TqBacktest, TargetPosTask, tafunc)
from utils.config_reader import ConfigReader
from utils.log import logger
from datetime import datetime, timedelta, date, time
from contextlib import closing

config_reader = ConfigReader('config/api.ini')
tqkq_user, tqkq_pwd = config_reader.read_config('tqkq', 'user'), config_reader.read_config('tqkq', 'pwd')
simnow_user, simnow_pwd = config_reader.read_config('simnow', 'user'), config_reader.read_config('simnow', 'pwd')
hyqh_user, hyqh_pwd = config_reader.read_config('hyqh', 'user'), config_reader.read_config('hyqh', 'pwd')

# api = TqApi(TqAccount("simnow", simnow_user, simnow_pwd), auth=TqAuth(tqkq_user, tqkq_pwd), web_gui=False)
api = TqApi(TqAccount("H宏源期货", hyqh_user, hyqh_pwd), auth=TqAuth(tqkq_user, tqkq_pwd), web_gui=False)

father_dir = os.path.abspath(os.path.dirname(os.getcwd()))  # 上一级目录
grandfather_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # 上上级目录
dl_path = os.path.join(father_dir, 'data', 'min_bar', 'tq\\')


async def download(symbol, duration_seconds, data_length):
    k_df = await api.get_kline_serial(symbol=symbol, duration_seconds=duration_seconds, data_length=data_length)
    async with api.register_update_notify() as update_chan:
        async for _ in update_chan:
            k_df['dt'] = k_df['datetime'].map(lambda x: tafunc.time_to_datetime(x))
            k_df = k_df.set_index(k_df['dt'])
            k_df = k_df.drop(columns=['dt'])
            k_df['date'] = k_df.index.map(lambda x: x.date())
            k_df['time'] = k_df.index.map(lambda x: x.time())
            k_df.to_csv(os.path.join(dl_path, f"{symbol}.csv"))


if __name__ == '__main__':
    code_list = [
        "SHFE.rb2205", "SHFE.bu2206", "SHFE.sp2205", "SHFE.ag2206", "SHFE.ss2204", "SHFE.zn2204", "SHFE.al2204",
        "CZCE.AP205", "CZCE.SF205", "CZCE.FG205", "CZCE.RM205", "CZCE.CJ205", "CZCE.SM205", "CZCE.PK204", "CZCE.MA205",
        "DCE.i2205", "DCE.jm2205", "DCE.v2205", "DCE.eg2205", "DCE.p2205", "DCE.y2205", "DCE.b2203", "DCE.pp2205"
    ]

    logger.info(f"下载期货品种个数：{len(code_list)}")

    for i in code_list:
        logger.info(f"开始下载, {i}")
        api.create_task(download(symbol=i, duration_seconds=60, data_length=8000))

    with closing(api):
        while True:
            api.wait_update()

