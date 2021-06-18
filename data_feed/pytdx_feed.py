from pytdx.reader import TdxDailyBarReader, TdxMinBarReader, TdxLCMinBarReader, TdxFileNotFoundException

# 通达信下载：https://www.tdx.com.cn/soft.html，选：金融终端V7.53
reader = TdxDailyBarReader()
df = reader.get_df("D:\\new_jyplug\\vipdoc\\sh\\lday\\sh600519.day")
reader = TdxLCMinBarReader()
df = reader.get_df("D:\\new_jyplug\\vipdoc\\ds\\fzline\\47#IC2106.lc5 ")
df1 = reader.get_df("D:\\new_jyplug\\vipdoc\\ds\\minline\\30#RB2110.lc1 ")
df.to_csv('IC2106_5min.csv')
df1.to_csv('RB2110_1min.csv')
print(df.tail())
print(df1.tail())

