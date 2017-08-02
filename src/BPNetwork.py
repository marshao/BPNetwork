#!/usr/local/bin/python
# coding: GBK

__metclass__ = type

import time, progressbar, pandas, urllib, re
import datetime
from sqlalchemy import create_engine, MetaData, text
from apscheduler.schedulers.background import BackgroundScheduler
import multiprocessing as mp
import logging
from math import pow, log


class C_BPNetwork:
    '''
    This class is designed to use Back Propagation Network to predict up and down of a specific stock.
    Creation Date: 20170802
    Author: GH
    '''

    def __init__(self):
        self._output_dir = 'D:\Personal\DataMining\\31_Projects\\01.Finance\\07.NN\BPNetwork\output\\'
        self._input_dir = 'D:\Personal\DataMining\\31_Projects\\01.Finance\\07.NN\BPNetwork\input\\'
        self._install_dir = 'D:\Personal\DataMining\\31_Projects\\01.Finance\\07.NN\BPNetwork\\'
        self._engine = create_engine('mysql+mysqldb://marshao:123@10.175.10.231/DB_StockDataBackTest')
        self._metadata = MetaData(self._engine)
        self._log_mesg = ''
        self._op_log = 'operLog.txt'
        self._x_min = ['m5', 'm15', 'm30', 'm60']
        self._x_period = ['day', 'week']

    def _time_tag(self):
        time_stamp_local = time.asctime(time.localtime(time.time()))
        time_stamp = datetime.datetime.now()
        only_date = time_stamp.date()
        return time_stamp

    def _time_tag_dateonly(self):
        time_stamp_local = time.asctime(time.localtime(time.time()))
        time_stamp = datetime.datetime.now()
        only_date = time_stamp.date()
        return only_date

    def _write_log(self, log_mesg, logPath='operLog.txt'):
        # logPath = str(self._time_tag_dateonly()) + logPath
        fullPath = self._output_dir + logPath
        if isinstance(log_mesg, str):
            with open(fullPath, 'a') as log:
                log.writelines(log_mesg)
        else:
            for message in log_mesg:
                with open(fullPath, 'a') as log:
                    log.writelines(message)
        self._log_mesg = ''

    def _clean_table(self, table_name):
        conn = self._engine.connect()
        conn.execute("truncate %s" % table_name)
        print "Table %s is cleaned" % table_name

    def _progress_monitor(self):
        '''
        Setting up progress bar to monitor the progress of whole program.
        '''
        widgets = ['MACD_Pattern_BackTest: ',
                   progressbar.Percentage(), ' ',
                   progressbar.Bar(marker='0', left='[', right=']'), ' ',
                   progressbar.ETA()]
        progress = progressbar.ProgressBar(widgets=widgets)
        return progress

    def _processes_pool(self, tasks, processors):
        # This is a self made Multiprocess pool
        task_total = len(tasks)
        loop_total = task_total / processors
        print "task total is %s, processors is %s, loop_total is %s" % (task_total, processors, loop_total)
        alive = True
        task_finished = 0
        # task_alive = 0
        # task_remain = task_total - task_finished
        # count = 0
        i = 0
        while i < loop_total:
            # print "This is the %s round" % i
            for j in range(processors):
                k = j + processors * i
                print "executing task %s" % k
                if k == task_total:
                    break
                tasks[k].start()
                j += 1

            for j in range(processors):
                k = j + processors * i
                if k == task_total:
                    break
                tasks[k].join()
                j += 1

            while alive == True:
                n = 0
                alive = False
                for j in range(processors):
                    k = j + processors * i
                    if k == task_total:
                        # print "This is the %s round of loop"%i
                        break
                    if tasks[k].is_alive():
                        alive = True
                    time.sleep(1)
            i += 1


class C_Sub_BuildUpNetwork(C_BPNetwork):
    def __init__(self):
        C_BPNetwork.__init__(self)

    def Build_Network_Main(self, layers=None, Inode=None, H1node=None, H2node=None, Onode=None):
        '''
        This is the main function to build up a Back Propagation Network
        :param layers: layer numbers of the network
        :param Inode: Number of input nodes
        :param H1node: Number of Hidden Layer 1 nodes
        :param H2node: Number of Hidden Layer 2 nodes
        :param Onode: Numbers of Output Layer nodes
        :return:
        '''
        if layers is None:
            layers = 3
        if Inode is None:
            Inode = 8
        if H1node is None:
            H1node = 5
        if H2node is None:
            H2node = 0
        if Onode is None:
            Onode = 4

    def _Build_Input_Layer(self, Inode):
        pass

    def _Build_Hidden_Layer(self, layers, H1node, H2node):
        pass

    def _Build_Output_Layer(self, Onode):
        pass

    def _Initialize_Coefs(self, network):
        pass

    def _Cal_Node_Output(self):
        pass

    def _Cal_Node_Activation(self):
        pass

    def _Cal_Cost_Function(self):
        pass


class C_Sub_TrainNetwork(C_BPNetwork):
    def __init__(self):
        C_BPNetwork.__init__(self)

    def Get_Training_Data(self):
        pass

    def Train_Main(self, mode=None):
        if mode is None:
            mode = 'B'  # Batch Train or Singal Train
        pass

    def _Forward_Calculation(self):
        pass

    def _Backward_Calculation(self):
        pass

    def _Save_Mode(self):
        pass


class C_Sub_NetworkPrediction(C_BPNetwork):
    def __init__(self):
        C_BPNetwork.__init__(self)

    def _Load_Mode(self):
        pass

    def Predict(self):
        pass


def main():
    pass


if __name__ == '__main__':
    main()
