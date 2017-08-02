#!/usr/local/bin/python
# coding: GBK

__metclass__ = type

import time, pandas, urllib, re
import datetime
from sqlalchemy import create_engine, Table, Column, MetaData
from sqlalchemy.sql import select, and_, or_, not_
from apscheduler.schedulers.background import BackgroundScheduler
import multiprocessing as mp
import logging
from math import pow


class C_GettingData:
    '''
    This is the class to monitor the stock real time price from various resources: (Sina, SnowBall, etc)
    '''

    def __init__(self):
        self._output_dir = 'D:\Personal\DataMining\\31_Projects\\01.Finance\\03.StockAutoTrades\output\\'
        self._input_dir = 'D:\Personal\DataMining\\31_Projects\\01.Finance\\03.StockAutoTrades\input\\'
        self._install_dir = 'D:\Personal\DataMining\\31_Projects\\01.Finance\\03.StockAutoTrades\\'

        self._data_source = {'sina': 'http://hq.sinajs.cn/list=', 'qq_realtime': 'http://qt.gtimg.cn/q=%s',
                             'qq_1_min': 'http://web.ifzq.gtimg.cn/appstock/app/minute/query?_var=min_data_%s&code=%s',
                             'qq_x_min': 'http://ifzq.gtimg.cn/appstock/app/kline/mkline?param=%s,%s,,%s',
                             'qq_x_period': 'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=%s,%s,,,%s,%s',
                             'qq_realtime_capital': 'http://stock.gtimg.cn/data/view/ggdx.php?t=2&q=%s'}
        self._x_min = ['m1', 'm5', 'm15', 'm30', 'm60']
        self._x_period = ['day', 'week']
        self._q_count = ['320', '50', '16', '8', '4']
        self._fq = ['qfq', 'hfq', 'bfq']
        self._stock_code = ['sz300226', 'sh600887', 'sz300146', 'sh600221']
        # self._stock_code = ['sz300226']
        self._log_mesg = ''
        self._op_log = 'operLog.txt'
        self._engine = create_engine('mysql+mysqldb://marshao:123@10.175.10.231/DB_StockDataBackTest')
        self._metadata = MetaData(self._engine)
        self._my_real_time_DF_columns_sina = ['stock_code', 'close_price', 'open_price', 'current_price', 'high_price',
                                              'low_price', 'buy_price', 'sale_price', 'trading_volumn',
                                              'trading_amount',
                                              'buy1_apply', 'buy1_price', 'buy2_apply', 'buy2_price', 'buy3_apply',
                                              'buy3_price', 'buy4_apply', 'buy4_price', 'buy5_apply', 'buy5_price',
                                              'sale1_apply', 'sale1_price', 'sale2_apply', 'sale2_price', 'sale3_apply',
                                              'sale3_price', 'sale4_apply', 'sale4_price', 'sale5_apply', 'sale5_price',
                                              'today', 'totime']
        self._real_time_DF_columns_qq = ['stock_name', 'stock_code', 'current_price', 'close_price', 'open_price',
                                         'current_buy_amount', 'current_sale_amount',
                                         'buy1_price', 'buy1_apply', 'buy2_price', 'buy2_apply', 'buy3_price',
                                         'buy3_apply',
                                         'buy4_price', 'buy4_apply', 'buy5_price', 'buy5_apply',
                                         'sale1_price', 'sale1_apply', 'sale2_price', 'sale2_apply', 'sale3_price',
                                         'sale3_apply',
                                         'sale4_price', 'sale4_apply', 'sale5_price', 'sale5_apply',
                                         'time', 'net_chg', 'net_chg_percent', 'high_price', 'low_price',
                                         'total_volumn', 'total_amount',
                                         'turnover_rate', 'PE', 'circulation_market_value', 'total_market_value', 'PB',
                                         'limit_up', 'limit_down']
        self._x_min_columns = ['quote_time', 'open_price', 'high_price', 'low_price', 'close_price', 'trading_volumn',
                               'stock_code', 'period']
        self._1_min_columns = ['quote_time', 'price', 'trading_volumn', 'stock_code']
        self._stock_minitue_data_DF = pandas.DataFrame(columns=self._my_real_time_DF_columns_sina)
        self._x_min_data_DF = pandas.DataFrame(columns=self._x_min_columns)
        self._x_period_data_DF = pandas.DataFrame(columns=self._x_min_columns)
        self._1_min_data_DF = pandas.DataFrame(columns=self._1_min_columns)
        self._real_time_data_DF = pandas.DataFrame(columns=self._real_time_DF_columns_qq)
        self._start_morning = datetime.time(9, 30, 0)
        self._end_morning = datetime.time(11, 42, 0)
        self._start_afternoon = datetime.time(13, 0, 0)
        self._end_afternoon = datetime.time(15, 12, 0)
        self._fun = self._empty_fun

    def service_getting_data(self):
        # get_data = self._timer(self.get_data_qq)
        last_run = time.time()
        while True:
            current = time.time()
            current_time = datetime.datetime.now().time()
            lag = current - last_run
            print "Tring to save data at last_run %s and current %s  and current - last_run %s" % (
                last_run, current, str(lag))
            time.sleep(15)
            if (current_time > self._start_morning and current_time < self._end_morning) or (
                            current_time > self._start_afternoon and current_time < self._end_afternoon):
                # Need a while true loop in here to keep hearing the real time data
                if current - last_run > 900:  # read data from web site at every 15 min
                    for stock in self._stock_code:
                        self.get_data_qq(stock, period='m5')
                        print 'saved m5 data'
                        time.sleep(5)
                        self.get_data_qq(stock, period='m1')
                        print 'saved m1 data'
                        time.sleep(5)
                        self.get_data_qq(stock, period='m30')
                        print 'saved m30 data'
                        time.sleep(5)
                        self.get_data_qq(stock, period='m60')
                        print 'saved m1 data'
                        time.sleep(5)
                        self.get_data_qq(stock, period='real')
                        print 'saved real time data'
                    last_run = time.time()
                    self._log_mesg = 'Write data to DB at ', self._time_tag()
                    print self._log_mesg
                    print "Saving data process took %s seconds" % (last_run - current)
                    # apply_pattern(17, 'm30', 'sz300226')
            else:
                print "Not in transaction time, wait 10 min to try again."
                st_time = datetime.time(21, 0, 0)
                ed_time = datetime.time(21, 15, 0)
                if (current_time > st_time) and (current_time < ed_time):
                    pass
                    # best_pattern_daily_calculate()
                time.sleep(600)

    def get_data_qq(self, stock_code='sz300226', period='day', fq='qfq', q_count='320'):
        logging.debug('Starting')
        self._stock_minitue_data_DF = pandas.DataFrame(columns=self._my_real_time_DF_columns_sina)
        self._x_min_data_DF = pandas.DataFrame(columns=self._x_min_columns)
        self._1_min_data_DF = pandas.DataFrame(columns=self._1_min_columns)
        got = True

        try:
            if period in self._x_period:  # precess day/week data
                fq = ('qfq' if fq not in (self._fq) else fq)
                url = self._data_source['qq_x_period'] % (stock_code, period, q_count, fq)
                html = urllib.urlopen(url)
                data = html.read()

                self._process_x_period_data_qq(data, period, fq, stock_code)

                self._save_data_to_db_qq(period, stock_code)
                print "Processed stock %s" % stock_code
            elif period in (self._x_min):
                if period == 'm1':  # process 1 min data
                    url = self._data_source['qq_1_min'] % (stock_code, stock_code)
                    html = urllib.urlopen(url)
                    data = html.read()
                    self._process_1_min_data_qq(data, stock_code)
                    self._save_data_to_db_qq(period, stock_code)
                else:  # process X min data
                    if period == 'm5':
                        q_count = self._q_count[1]
                        url = self._data_source['qq_x_min'] % (stock_code, period, q_count)
                    elif period == 'm15':
                        q_count = self._q_count[2]
                        url = self._data_source['qq_x_min'] % (stock_code, period, q_count)
                    elif period == 'm30':
                        q_count = self._q_count[3]
                        url = self._data_source['qq_x_min'] % (stock_code, period, q_count)
                    elif period == 'm60':
                        q_count = self._q_count[3]
                        url = self._data_source['qq_x_min'] % (stock_code, period, q_count)
                    else:
                        q_count = self._q_count[4]
                        url = self._data_source['qq_x_min'] % (stock_code, period, q_count)
                    html = urllib.urlopen(url)
                    data = html.read()
                    self._process_x_min_data_qq(data, period, stock_code)
                    self._save_data_to_db_qq(period, stock_code)
            else:  # process real time data
                url = self._data_source['qq_realtime'] % stock_code
                html = urllib.urlopen(url)
                data = html.read()
                self._process_real_time_data_qq(data)
                self._save_data_to_db_qq(period, stock_code)
        except:
            got = False
        logging.debug('Finished')
        return got

    def _process_x_period_data_qq(self, data, period, fq, stock_code):
        self._x_period_data_DF.drop(self._x_period_data_DF.index[:], inplace=True)
        sPos = data.find(fq + period) + len(fq + period) + 4
        ePos = data.find('"qt"')
        data = data[sPos:(ePos - 3)].split('],[')
        p = re.compile(r'\d+.\d+.\d+')

        for item in data:
            if item.find('{"nd"') != -1:
                item = item[:item.find('{"nd"')]
            tmp_l = p.findall(item)
            # tmp_l[0] = re.sub(r'[^\d]', '', tmp_l[0])
            tmp_l[0] = datetime.datetime.strptime(tmp_l[0], '%Y-%m-%d')
            i = 1
            while i < 6:
                try:
                    if tmp_l[i] != '':
                        tmp_l[i] = float(tmp_l[i])
                except:  # the except in here is to deal with the today's value, which is missing high, low, close
                    tmp_v = tmp_l[i - 1]
                    tmp_l[i - 1] = 0.00
                    tmp_l.append(tmp_v)
                i += 1
            tmp_l.append(stock_code)
            tmp_l.append(period)
            # print tmp_l

            self._x_period_data_DF.loc[len(self._x_period_data_DF)] = tmp_l
        # print self._x_min_data_DF
        self._x_period_data_DF.set_index('quote_time', inplace=True)
        # print self._x_min_data_DF
        return self._x_period_data_DF

    def _process_real_time_data_qq(self, data):
        self._real_time_data_DF.drop(self._real_time_data_DF.index[:], inplace=True)
        sPos = data.find('=')
        data = data[sPos + 2:-3].split('~')
        tmp_l = []
        tmp_l.append(data[1])  # stock_name
        tmp_l.append(data[2])  # stock_code
        tmp_l.append(float(data[3]))  # current_price
        tmp_l.append(float(data[4]))  # close_price
        tmp_l.append(float(data[5]))  # open_price
        tmp_l.append(int(data[7]))  # buying_amount
        tmp_l.append(int(data[8]))  # saling_amount
        tmp_l.append(float(data[9]))  # buy1_price
        tmp_l.append(int(data[10]))  # buy1_amount
        tmp_l.append(float(data[11]))  # buy2_price
        tmp_l.append(int(data[12]))  # buy2_amount
        tmp_l.append(float(data[13]))  # buy3_price
        tmp_l.append(int(data[14]))  # buy3_amount
        tmp_l.append(float(data[15]))  # buy4_price
        tmp_l.append(int(data[16]))  # buy4_amount
        tmp_l.append(float(data[17]))  # buy5_price
        tmp_l.append(int(data[18]))  # buy5_amount
        tmp_l.append(float(data[19]))  # sale1_price
        tmp_l.append(int(data[20]))  # sale1_amount
        tmp_l.append(float(data[21]))  # sale2_price
        tmp_l.append(int(data[22]))  # sale2_amount
        tmp_l.append(float(data[23]))  # sale3_price
        tmp_l.append(int(data[24]))  # sale3_amount
        tmp_l.append(float(data[25]))  # sale4_price
        tmp_l.append(int(data[26]))  # sale4_amount
        tmp_l.append(float(data[27]))  # sale5_price
        tmp_l.append(int(data[28]))  # sale5_amount
        tmp_l.append(datetime.datetime.strptime(data[30], '%Y%m%d%H%M%S'))  # trade_time
        tmp_l.append(float(data[31]))  # net_chg
        tmp_l.append(float(data[32]))  # net_chg_percent
        tmp_l.append(float(data[33]))  # high_price
        tmp_l.append(float(data[34]))  # low_price
        tmp_l.append(int(data[36]))  # trading_volumn
        tmp_l.append(int(data[37]))  # trading_amount
        tmp_l.append(float(data[38]))  # turnover_rate
        if data[39] != '':
            tmp_l.append(float(data[39]))  # PE
        else:
            tmp_l.append(0.00)  # PE
        tmp_l.append(float(data[44]))  # circulate_market_value
        tmp_l.append(float(data[45]))  # total_market_value
        tmp_l.append(float(data[46]))  # PB
        tmp_l.append(float(data[47]))  # limit_up
        tmp_l.append(float(data[48]))  # limit_down
        self._real_time_data_DF.loc[len(self._real_time_data_DF)] = tmp_l
        print "processed real time data"
        return self._real_time_data_DF

    def _process_1_min_data_qq(self, data, stock_code):
        self._1_min_data_DF.drop(self._1_min_data_DF.index[:], inplace=True)
        today = datetime.date.today().strftime('%Y-%m-%d')
        sPos = data.find('{"data":[')
        ePos = data.find('"date":')
        data = data[(sPos + 9):(ePos - 2)].split(',')
        p = re.compile(r'\d+\.*\d+')
        for item in data:
            tmp_l = p.findall(item)
            tmp_l[0] = today + ' ' + tmp_l[0]
            tmp_l[0] = datetime.datetime.strptime(tmp_l[0], '%Y-%m-%d %H%M')
            tmp_l[1] = float(tmp_l[1])
            tmp_l[2] = float(tmp_l[2])
            tmp_l.append(stock_code)
            self._1_min_data_DF.loc[len(self._1_min_data_DF)] = tmp_l
        self._1_min_data_DF.set_index('quote_time', inplace=True)
        print "processed 1 min data of %s" % stock_code
        return self._1_min_data_DF

    def _process_x_min_data_qq(self, data, x_min, stock_code):
        self._x_min_data_DF.drop(self._x_min_data_DF.index[:], inplace=True)
        # print self._x_min_data_DF
        nPos = data.find(x_min)
        data = data[(nPos + 7):-5].split('],[')
        p = re.compile(r'\d+.\d+')
        for item in data:
            tmp_l = p.findall(item)
            tmp_l[0] = datetime.datetime.strptime(tmp_l[0], '%Y%m%d%H%M')
            i = 1
            while i < 6:
                tmp_l[i] = float(tmp_l[i])
                i += 1
            tmp_l.append(stock_code)
            tmp_l.append(x_min)
            # print len(self._x_min_data_DF)
            self._x_min_data_DF.loc[len(self._x_min_data_DF)] = tmp_l
            # print self._x_min_data_DF['quote_time']
        self._x_min_data_DF.set_index('quote_time', inplace=True)
        print "processed %s data of stock %s" % (x_min, stock_code)
        return self._x_min_data_DF

    def _save_data_to_db_qq(self, period, stock_code):
        # save today's data (1min, 5min, 30min), min data will be downloaded and saved into DB in every 30 min from 9:30
        # Data will be exanmed to avoid inserting duplicate rows into DB.
        # Columns: stock_code, period, quote_time will be used to exanmed the duplicated row.
        if period in self._x_min:
            # data = self._x_min_data_DF
            # save x min data into DB
            data = self._remove_duplicate_rows(period, stock_code)
            if period != 'm1':
                data.to_sql('tb_StockXMinRecords', self._engine, if_exists='append', index=True)
                print "saved %s period data of stock code %s at time %s" % (period, stock_code, self._time_tag())
            else:
                data.to_sql('tb_Stock1MinRecords', self._engine, if_exists='append', index=True)
                print "saved m1 data of stock code %s at %s" % (stock_code, self._time_tag())
        elif period in self._x_period:
            # data = self._x_period_data_DF
            # save historical data (day, week), historical data will be downloaded and saved into DB when it is required.
            data = self._remove_duplicate_rows(period, stock_code)
            data.to_sql('tb_StockXPeriodRecords', self._engine, if_exists='append', index=True)
            print "saved %s period data of stock code %s at %s" % (period, stock_code, self._time_tag())
        else:
            # save real_time data, real time data will be downloaded and saved into DB in every 2 sec.
            data = self._real_time_data_DF
            data.to_sql('tb_StockRealTimeRecords', self._engine, if_exists='append', index=False)

    def _remove_duplicate_rows(self, period, stock_code):
        '''
        Step 1: 从数据库中取出相应股票最后一个记录的时间戳
        Step 2：从data 删除早于这个时间戳的记录
        Step 3: 返回清理后的DataFrame
        :param df:
        :return:
        '''
        # print "Jump into remove duplication"
        conn = self._engine.connect()
        if period in self._x_min:  # process X min data
            if period != 'm1':
                tb = Table('tb_StockXMinRecords', self._metadata, autoload=True)
                s = select([tb.c.quote_time]). \
                    where(
                    and_(tb.c.period == period,
                         tb.c.stock_code == stock_code)). \
                    order_by(tb.c.quote_time.desc()).limit(1)
                data = self._x_min_data_DF
            else:  # Process 1 min data
                tb = Table('tb_Stock1MinRecords', self._metadata, autoload=True)
                s = select([tb.c.quote_time]). \
                    where(tb.c.stock_code == stock_code). \
                    order_by(tb.c.quote_time.desc()).limit(1)
                data = self._1_min_data_DF
        else:  # process day / week data
            tb = Table('tb_StockXPeriodRecords', self._metadata, autoload=True)
            s = select([tb.c.quote_time]). \
                where(
                and_(tb.c.period == period,
                     tb.c.stock_code == stock_code)). \
                order_by(tb.c.quote_time.desc()).limit(1)
            data = self._x_period_data_DF

        result = conn.execute(s).fetchall()
        if len(result) != 0:
            # print result
            last_record_time = result[0][0]
            # print last_record_time
            # data = data.loc[lambda df: df.index > last_record_time, :]
            data = data[data.index > last_record_time]
            return data
        else:
            return data

    def _timer(self, func):
        # 定义一个计时器函数，让get_real_time_data 每60秒向数据库传送一次更新的数据。
        # 定义一个内嵌的包装函数，给传入的函数加上计时功能的包装
        def wrapper():
            start = time.time()
            while (time.time() - start) < 900:  # call function at every 15 min
                func(None, None, None, None)
                end = time.time()
                time.sleep(5)
                print "time spent of each run = ", (end - start)

        return wrapper

    def _time_tag(self):
        time_stamp_local = time.asctime(time.localtime(time.time()))
        time_stamp = datetime.datetime.now()
        only_date = time_stamp.date()
        return time_stamp_local

    def _time_tag_dateonly(self):
        time_stamp_local = time.asctime(time.localtime(time.time()))
        time_stamp = datetime.datetime.now()
        only_date = time_stamp.date()
        return only_date

    def _write_log(self, log_mesg, logPath):
        fullPath = self._output_dir + logPath
        with open(fullPath, 'a') as log:
            log.writelines(log_mesg)

    def _convert_encoding(self, lines, new_coding='UTF-8'):
        try:
            encoding = 'GB2312'
            data = lines.decode(encoding)
            data = data.encode(new_coding)
        except:
            data = 'DecodeError'
        return data

    def _empty_fun(self, period):
        pass

    def job_schedule(self):
        # job_stores = {'default': MemoryJobStore()}
        # executor = {'processpool': ThreadPoolExecutor(8)}
        # job_default = {'coalesce': False, 'max_instances': 12}
        # scheduler_1 = BackgroundScheduler(jobstores=job_stores, executors=executor, job_defaults=job_default)

        scheduler_1 = BackgroundScheduler()
        # scheduler_2 = BlockingScheduler()
        # scheduler_1.add_job(self._fun, 'interval', seconds=180, args=['m1'])
        # scheduler_1.add_job(self._fun, 'interval', seconds=300, args=['m5'])
        # scheduler_1.add_job(self._fun, 'interval', seconds=1800, args=['m30'])
        scheduler_1.add_job(self._data_service, 'interval', seconds=180, args=['m1'])
        scheduler_1.add_job(self._data_service, 'interval', seconds=300, args=['m5'])
        scheduler_1.add_job(self._data_service, 'interval', seconds=3600, args=['m60'])
        scheduler_1.add_job(self._half_hour_tasks, 'interval', seconds=1800, args=['m30'])
        scheduler_1.start()

        # The switch of scheduler_1
        while True:
            self._scheduler_switch(scheduler_1)

    def _half_hour_tasks(self, period):
        self._data_service(period)
        # apply_pattern('m30', 'sz300226')

    def _load_base_finance_value(self):
        pass

    def _scheduler_switch(self, scheduler):
        current_time = datetime.datetime.now().time()

        if (current_time > self._start_morning and current_time < self._end_morning) or (
                        current_time > self._start_afternoon and current_time < self._end_afternoon):
            scheduler.resume()
            print "scheduler back to work"
            # time.sleep(600)
        else:
            print "out of the time of getting data"
            scheduler.pause()
            st_time = datetime.time(21, 0, 0)
            ed_time = datetime.time(21, 15, 0)
            if (current_time > st_time) and (current_time < ed_time):
                pass
                # best_pattern_daily_calculate()
        time.sleep(600)

    def _data_service(self, period):
        processes = []
        for stock in self._stock_code:
            p = mp.Process(target=self.get_data_qq, args=(stock, period, 'qfq',))
            processes.append(p)
        for p in processes:
            p.start()
        for p in processes:
            p.join()


class C_GettingSVMData(C_GettingData):
    def __init__(self):
        C_GettingData.__init__(self)

        self._base_finance_value = {'sz300226': [['2017-03-31', 0.0369, 3.19, 147709300, 0.1621],
                                                 ['2016-12-31', 0.1402, 3.13, 147794500, 0.1594],
                                                 ['2016-09-30', 0.1019, 3.07, 147794500, -0.5266],
                                                 ['2016-06-30', 0.0871, 2.47, 147794500, -1.2731],
                                                 ['2016-03-31', 0.0258, 2.32, 10862390, -2.2687],
                                                 ['2015-12-31', -1.61, 0.5, 102955400, -2.4462],
                                                 ['2015-09-30', -1.1795, 0.92, 102955400, -1.6715],
                                                 ['2015-06-30', -0.6977, 1.4, 102955400, -0.8785]],
                                    'sh600867': [['2017-03-31', 0.15, 2.91, 1363302500, 0.0365],
                                                 ['2016-12-31', 0.46, 2.77, 1363302500, 0.1083],
                                                 ['2016-09-30', 0.36, 2.64, 1360093300, 0.0971],
                                                 ['2016-06-30', 0.23, 1.86, 1357915300, 0.0727],
                                                 ['2016-03-31', 0.14, 2.31, 1357915300, 0.0843],
                                                 ['2015-12-31', 0.43, 2.17, 1131596100, 0.1743],
                                                 ['2015-09-30', 0.34, 2, 05, 1131596100, 0.1415],
                                                 ['2015-06-30', 0.23, 1.92, 1127060600, 0.0982]],
                                    'sz002310': [['2017-03-31', -0.01, 3.42, 1419758100, 0.4088],
                                                 ['2016-12-31', 0.51, 3.43, 1419758100, 0.1241],
                                                 ['2016-09-30', 0.20, 2.66, 1416089000, 0.1841],
                                                 ['2016-06-30', 0.29, 6.43, 566435600, 0.2183],
                                                 ['2016-03-31', -0.07, 6.13, 547201800, 0.0022],
                                                 ['2015-12-31', 0.6, 6.20, 558157300, -0.0895],
                                                 ['2015-09-30', 0.26, 5.84, 554991100, -0.2729],
                                                 ['2015-06-30', 0.16, 5.74, 554991100, -0.3406]],
                                    'sh600221': [['2017-03-31', 0.0497, 3.08, 11812064200, -0.0513],
                                                 ['2016-12-31', 0.21, 3.03, 11812064200, 0.0115],
                                                 ['2016-09-30', 0.27, 3.03, 11812064200, 0.0773],
                                                 ['2016-06-30', 0.137, 2.68, 11812064200, 0.006],
                                                 ['2016-03-31', 0.1183, 2.65, 11812064200, 0.044],
                                                 ['2015-12-31', 0.2460, 2.54, 11812064200, 0.0348],
                                                 ['2015-09-30', 0.2040, 2.46, 11812064200, 0.2108],
                                                 ['2015-06-30', 0.1315, 2.43, 11812064200, 0.1357]],
                                    'sz300146': [['2017-03-31', 0.1944, 3.11, 871425000, 0.0716],
                                                 ['2016-12-31', 0.37, 3.18, 860141000, -0.1166],
                                                 ['2016-09-30', 0.39, 3.23, 860141000, 0.0625],
                                                 ['2016-06-30', 0.25, 3.08, 860141000, 0.3205],
                                                 ['2016-03-31', 0.3088, 5.97, 860141000, 0.7921],
                                                 ['2015-12-31', 0.89, 3.16, 359358600, 0.3883],
                                                 ['2015-09-30', 0.82, 6.2, 334988400, 1.7371],
                                                 ['2015-06-30', 0.59, 5.95, 337432500, 1.2362]]
                                    }
        # 日期，稀释每股收益，每股净资产，流通A股， EPS(TTM)
        self._base_finance_value_columns = ['quote_time', 'diluted_eps', 'BVPS', 'A_total', 'EPS_TTM']

    def get_named_minitue_capital(self, stock_code=None, time=None, df=None):
        '''
        Look for capital data of a stock at given time
        :param stock_code: stock code 'sh600867'
        :param time: given time '14:55:00'
        :param df: The DF of the named minute data
        :return: a float value of 主力 capital
        '''
        if stock_code is None:
            stock_code = 'sh600867'
        if time is None:
            time = '14:55:00'
        if df is None:
            print 'The parameter DF is missing.'
            return
        url = self._data_source['qq_realtime_capital'] % (stock_code)
        html = urllib.urlopen(url)
        data = html.read()
        # locate the position of given time
        sPos = data.find(time)
        # The wanted data is between the 5th and 4th ~ in reverse order start from the given time position
        i = 0
        while i < 5:
            if i == 0:
                sPos = data.rfind('~', 0, sPos)
            else:
                sPos = data.rfind('~', 0, (sPos - 2))

            if i == 3: ePos = sPos
            i += 1
        capital = data[sPos + 1: ePos]
        df.set_value(time, 'Capital', float(capital))
        return df

    def get_daily_capital(self, stock_code=None, days=None):
        '''
        Look for capital data of a stock at given time
        :param stock_code: stock code 'sh600867'
        :param time: given time '14:55:00'
        :param df: The DF of the named minute data
        :return: a float value of 主力 capital
        '''
        if stock_code is None:
            stock_code = 'sh600867'
        if days is None:
            days = '300'
        else:
            days = str(days)
        daily_capital = []
        url = 'http://stock.gtimg.cn/data/view/ggdx.php?t=3&q=%s&d=%s' % (stock_code, days)
        html = urllib.urlopen(url)
        data = html.read()
        # locate the position of =
        sPos = data.find('=')
        data = data[sPos + 2:-2].split('^')
        for each_day in data:
            daily = each_day.split('~')
            tmp = [daily[3], daily[1]]
            daily_capital.append(tmp)
        return daily_capital

    def get_named_minitue_price(self, stock_code=None, time=None):
        '''
        Look for price data of a stock at given time
        :param stock_code: stock code 'sh600867'
        :param time: given time '14:55:00'
        :return: a float value of price
        '''
        if stock_code is None:
            stock_code = 'sh600867'
        if time is None:
            time = '14:55:00'
        # Update minite data into DB.
        pp = C_GettingData()
        pp.get_data_qq(stock_code=stock_code, period='m1')
        df = self.load_data_from_db(stock_code=stock_code, start=time, period='m1', row_count=1)
        df.rename(columns={'price': 'close_price'}, inplace=True)
        return df

    def load_data_from_db(self, stock_code=None, start=None, end=None, period=None, row_count=None):
        '''
        Load Data from DB
        :param stock_code:
        :param start:
        :param end:
        :param period:
        :param row_count:
        :return:
        '''
        if stock_code is None:
            stock_code = 'sz300226'
        if period is None:
            period = 'day'
        if row_count is None:
            row_count = 300
        sql_fetch_batch_min_records = (
            "select * from tb_Stock1MinRecords where period = %s and stock_code = %s ORDER by quote_time DESC limit %s")
        sql_fetch_named_min_records = (
            "select * from tb_Stock1MinRecords where stock_code = %s and quote_time = %s limit %s")
        sql_fetch_period_records = (
            "select * from tb_StockXPeriodRecords where period = %s and stock_code = %s ORDER by quote_time DESC limit %s")
        if period == 'day' or period == 'week':
            sql_fetch_records = sql_fetch_period_records
            df_stock_records = pandas.read_sql(sql_fetch_records, con=self._engine, index_col='quote_time',
                                               params=(period, stock_code, row_count))
        elif row_count == 1:
            time_stamp = datetime.datetime.now()
            quote_date = time_stamp.date()
            # Using start parameter to transfer minitue
            quote_time = str(quote_date) + " " + start
            sql_fetch_records = sql_fetch_named_min_records
            df_stock_records = pandas.read_sql(sql_fetch_records, con=self._engine, index_col='quote_time',
                                               params=(stock_code, quote_time, row_count))
        else:
            sql_fetch_records = sql_fetch_batch_min_records
            df_stock_records = pandas.read_sql(sql_fetch_records, con=self._engine, index_col='quote_time',
                                               params=(period, stock_code, row_count))

        # df_stock_records = pandas.read_sql(sql_fetch_records, con=self._engine,params=(period, stock_code, row_count))
        return df_stock_records

    def load_capital_from_file_into_df(self, df_stock_records=None, source_file=None):
        '''
        Load data from file, (Captial 资金 data)
        :param df_stock_records:
        :return:
        '''
        if df_stock_records is None:
            return

        if source_file is None:
            source_file = 'Capital.csv'

        fn = open(source_file, 'r')
        for line in fn:
            line = line[:-1]  # Remove the /r/n
            vlist1 = line.split("/r")
            if vlist1[0] == "": continue
            vlist = vlist1[0].split(",")
            quote_time = datetime.datetime.strptime(vlist[0], '%Y-%m-%d')
            if quote_time in df_stock_records.index:
                df_stock_records.set_value(quote_time, 'Capital', float(vlist[1]))
            else:
                continue
        fn.close()
        return df_stock_records

    def load_captical_from_list_into_df(self, df_stock_records=None, capital_list=None):
        '''
        Load Daily Capital data into DF
        :param df_stock_records:
        :param capital_list: List with Capital data
        :return:
        '''
        for item in capital_list:
            quote_time = datetime.datetime.strptime(item[0], '%Y-%m-%d')
            if quote_time in df_stock_records.index:
                df_stock_records.set_value(quote_time, 'Capital', float(item[1]))
            else:
                continue
        df_stock_records.dropna(inplace=True)
        return df_stock_records

    def data_normalization(self, df_records=None):
        '''
        This is the function to normalize data into [0-1], using Z normal method.
        z-score(zero-mean normalization): x=(x-mean)/std
        The data source should be a panda data frame
        The data normalization should be done before any calculation on data
        :param df_records:This is the data frame waiting to be normalized.
        :return: new DF with normalized data
        '''
        means = df_records.mean(numeric_only=True)
        stds = df_records.std(numeric_only=True)

        for column in df_records:
            if isinstance(df_records[column][0], float):
                column_name = 'nor_' + column
                df_records[column_name] = (df_records[column] - means[column]) / stds[column]
            else:
                continue
        return df_records, means, stds

    def _cal_PBPE(self, stock_code, df_daily_record=None, ):
        '''
        Calculation of PB and PE_TTM
        PE_TTM = Stock_Price/EPS_TTM
        PB = Stock_Price/BVPS
        :param df_daily_record:
        :return:
        '''

        base_finance_value = self._base_finance_value[stock_code]
        if df_daily_record is None: return
        if len(base_finance_value) == 0: self._load_base_finance_value()
        df_daily_record['PE_TTM'] = 0.0
        df_daily_record['PB'] = 0.0
        i = 0
        for i in range(len(base_finance_value)):
            quote_time = datetime.datetime.strptime(base_finance_value[i][0], '%Y-%m-%d')
            diluted_eps = base_finance_value[i][1]
            BVPS = base_finance_value[i][2]
            A_total = base_finance_value[i][3]
            EPS_TTM = base_finance_value[i][4]
            if i != 0:
                quote_last_time = datetime.datetime.strptime(base_finance_value[i - 1][0], '%Y-%m-%d')
                df = df_daily_record.loc[
                    (df_daily_record.index >= quote_time) & (df_daily_record.index < quote_last_time)]
                # print df
                for index, row in df.iterrows():
                    PE_TTM = row['close_price'] / EPS_TTM
                    PB = row['close_price'] / BVPS
                    df_daily_record.set_value(index, 'PE_TTM', PE_TTM)
                    df_daily_record.set_value(index, 'PB', PB)
            else:
                # print df_daily_record.index
                df = df_daily_record.loc[(df_daily_record.index >= quote_time)]
                for index, row in df.iterrows():
                    PE_TTM = row['close_price'] / EPS_TTM
                    PB = row['close_price'] / BVPS
                    df_daily_record.set_value(index, 'PE_TTM', PE_TTM)
                    df_daily_record.set_value(index, 'PB', PB)
            i += 1

        df_daily_record = df_daily_record.loc[df_daily_record['PB'] != 0]

        # print df_daily_record
        return df_daily_record

    def _cal_StockAmp(self, stock_code, df_daily_record):
        '''
        This is to calculate the Amplitude 振幅
        :param stock_code:
        :param df_daily_record:
        :return:
        '''
        df_daily_record['SA'] = 0.0
        # df_iter = df_daily_record.iterrows()

        for i in range(0, (df_daily_record.shape[0] - 1)):
            # df_daily_record.set_value(index, 'PB',PB)
            # print  df_daily_record
            index = df_daily_record.index.values[i]
            SA = (df_daily_record.iloc[i]['high_price'] - df_daily_record.iloc[i]['low_price']) / \
                 df_daily_record.iloc[i + 1]['close_price']
            df_daily_record.set_value(index, 'SA', SA)
        return df_daily_record

    def _cal_TurnOver(self, stock_code, df_daily_record):
        base_finance_value = self._base_finance_value[stock_code]
        if df_daily_record is None: return
        if len(base_finance_value) == 0: self._load_base_finance_value()
        df_daily_record['Turn_Over'] = 0.0
        i = 0

        for i in range(len(base_finance_value)):
            quote_time = datetime.datetime.strptime(base_finance_value[i][0], '%Y-%m-%d')
            A_total = base_finance_value[i][3]
            if i != 0:
                quote_last_time = datetime.datetime.strptime(base_finance_value[i - 1][0], '%Y-%m-%d')
                df = df_daily_record.loc[
                    (df_daily_record.index >= quote_time) & (df_daily_record.index < quote_last_time)]
                # print df
                for index, row in df.iterrows():
                    Turn_Over = row['trading_volumn'] / A_total
                    df_daily_record.set_value(index, 'Turn_Over', Turn_Over)
            else:
                # print df_daily_record.index
                df = df_daily_record.loc[(df_daily_record.index >= quote_time)]
                for index, row in df.iterrows():
                    Turn_Over = row['trading_volumn'] / A_total
                    df_daily_record.set_value(index, 'Turn_Over', Turn_Over)
            i += 1
        return df_daily_record

    def _add_higher_degree_parameters(self, df, degree=2):
        '''
        Add higher order parameters in to df
        :param df:
        :return:
        '''
        for column in df:
            if column != 'id_tb_StockXPeriodRecords' and isinstance(df[column][0], str) is False:
                # pow(column, 2)
                new_column = column + '-Order%s' % degree
                # In here, df[column] is a series, df.pow will not be working
                df[new_column] = df[column].pow(degree)
        return df


def main():
    pp = C_GettingData()

    stock_code = ['sh600221', 'sh600867', 'sz002310', 'sz300146']

    time = '14:55:00'
    dimension = 'H'
    ps = C_GettingSVMData()
    # get_batch_svm_data('sh600221', pp, ps, dimension)
    for stock in stock_code:
        # get_batch_svm_data(stock, pp, ps, dimension)
        # -------------------------------------------------------
        get_named_minitue_svm_data(stock, pp, ps, time, dimension)
        # -----------------------------------------------------------


def get_batch_svm_data(stock_code, pp, ps, dimension=None):
    if dimension is None:
        dimension = 'L'
    pp.get_data_qq(stock_code=stock_code, period='day')
    df = ps.load_data_from_db(stock_code=stock_code, period='day')
    ls = ps.get_daily_capital(stock_code=stock_code, days=300)
    # df = ps.load_capital_from_file_into_df(df, source_file='input\\600867Capital.csv')
    df = ps.load_captical_from_list_into_df(df, ls)
    df = ps._cal_PBPE(stock_code, df)
    # -----------------------------------------
    # df = ps._cal_StockAmp(stock_code, df)
    # df = ps._cal_TurnOver(stock_code, df)
    # ------------------------------------------
    if dimension == 'H':
        df = ps._add_higher_degree_parameters(df)
        # df = ps._add_higher_degree_parameters(df, degree=3)
    df, means, stds = ps.data_normalization(df)
    df.to_csv('/home/marshao/SVMClassifier/webdata/stock_data_%s_%sO.csv' % (stock_code, dimension), header=True)


def get_named_minitue_svm_data(stock_code, pp, ps, time, dimension=None):
    if dimension is None:
        dimension = 'H'
    df = ps.get_named_minitue_price(stock_code=stock_code, time=time)
    df = ps.get_named_minitue_capital(stock_code=stock_code, time=time, df=df)
    df = ps._cal_PBPE(stock_code, df)
    # -----------------------------------------
    # df = ps._cal_StockAmp(stock_code, df)
    # df = ps._cal_TurnOver(stock_code, df)
    # -----------------------------------------
    if dimension == 'H':
        df = ps._add_higher_degree_parameters(df)
        cols_to_write = ['nor_close_price', 'nor_trading_volumn', 'nor_Capital', 'nor_PE_TTM', 'nor_PB',
                         'nor_close_price-Order2', 'nor_trading_volumn-Order2', 'nor_Capital-Order2',
                         'nor_PE_TTM-Order2', 'nor_PB-Order2']
    else:
        cols_to_write = ['nor_close_price', 'nor_trading_volumn', 'nor_Capital', 'nor_PE_TTM', 'nor_PB']
    # Data normalization using batch means and stds
    means, stds = get_batch_svm_data_mean_stds(stock_code, pp, ps)
    for column in df:
        if isinstance(df[column][0], float):
            column_name = 'nor_' + column
            df[column_name] = (df[column] - means[column]) / stds[column]
        else:
            continue
    df[cols_to_write].to_csv('/home/marshao/SVMClassifier/PredictFiles/stock_data_%s.csv' % stock_code, header=False,
                             index=False)


def get_batch_svm_data_mean_stds(stock_code, pp, ps):
    pp.get_data_qq(stock_code=stock_code, period='day')
    df = ps.load_data_from_db(stock_code=stock_code, period='day')
    ls = ps.get_daily_capital(stock_code=stock_code, days=300)
    # df = ps.load_capital_from_file_into_df(df, source_file='input\\600867Capital.csv')
    df = ps.load_captical_from_list_into_df(df, ls)
    df = ps._cal_PBPE(stock_code, df)
    # -----------------------------------------
    # df = ps._cal_StockAmp(stock_code, df)
    # df = ps._cal_TurnOver(stock_code, df)
    # -----------------------------------------
    df = ps._add_higher_degree_parameters(df)
    df, means, stds = ps.data_normalization(df)
    return means, stds


# load

if __name__ == '__main__':
    main()
