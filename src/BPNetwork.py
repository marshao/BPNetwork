#!/usr/local/bin/python
# coding: GBK

__metclass__ = type

import time, progressbar, urllib, re
import pandas as pd
import numpy as np
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

    def build_network_main(self, layers=None, inode=None, h1node=None, h2node=None, onode=None, batch_mode=True, momentum=False):
        '''
        This is the main function to build up a Back Propagation Network
        :param layers: layer numbers of the network
        :param inode: Number of input nodes
        :param h1node: Number of Hidden Layer 1 nodes
        :param h2node: Number of Hidden Layer 2 nodes
        :param onode: Numbers of Output Layer nodes
        :return:
        '''
        if layers is None:
            layers = 3
        elif layers != 3 and layers != 4:
            print "System can only  build 3 or 4 layers network."
            return

        if inode is None:
            inode = 8
        if h1node is None:
            h1node = 5
        if h2node is None:
            h2node = 0
        if onode is None:
            onode = 4

        network = {
            'layer':4,
            'inode':inode,
            'h1node':h1node,
            'h2node':h2node,
            'onode':onode,
            'H1ParameterNumbers':0,
            'H1theta':[], #Hidden Layer 1 Parameters
            'H2ParameterNumbers':0,
            'H2theta':[], #Hidden Layer 2 Parameters
            'OParameterNumbers':0,
            'Otheta':[], # Output layer Parameters
            'BatchMode':batch_mode, # Switch of Batch Mode
            'BCost':0.0, #batch Cost for whole Training Set
            'SCost':0.0, #Single Cost for each Training sample
            'CT':0.0, #Cost Threshold, the traget level of cost
            'CRound':0, #Current Training Rounds
            'MRound':0, #Max Training Rounds
            'lambada':0.0, #Learning Rate
            'epsilon':0.0, # Step length of Calculating the Derivative of parameter theta
            'MomentumMode':momentum, # Switch of Momentum Model
            'alpha': 0.0,  # momentum Factor
        }
        # Initialize the Parameters for each node.
        # The first columns of each set of parameter are 1 corresponding to the Bias.
        self._initialize_parameters(network)
        # Initialize the variation length of Theta, normally use 2epsilon to calculate the two sides derivative.
        network['epsilon'] = 0.012
        # Set the allowed Max Rounds of Training:
        network['MRound'] = 3000
        # Initialize the learning Rate
        network['lambada'] = 3
        # Setting up Momentum factor alpha:
        if network['MomentumMode']:
            network['alpha'] = 0.3



    def _Build_Input_Layer(self, Inode):
        pass

    def _Build_Hidden_Layer(self, layers, H1node, H2node):
        pass

    def _Build_Output_Layer(self, Onode):
        pass

    def _initialize_parameters(self, network):
        '''
        This fucntion is to initialize origin prameters for the network.
        The parameters are randoms selected number between 0 and 1.
        The parameters should be small enough
        :param network:
        :return:
        '''
        #H1theta = pd.DataFrame(np.random.random_sample((network['H1node'], network['Inode'])))
        H1theta = np.random.random_sample((network['H1node'],network['Inode']+1))/10
        H1theta[:, 0]=1
        network['H1theta']=H1theta
        if network['H2node'] != 0:
            H2theta = np.random.random_sample((network['H2node'], network['H1node'] + 1)) / 10
            H2theta[:, 0] = 1
            network['H2theta'] = H2theta
            Otheta = np.random.random_sample((network['Onode'], network['H2node'] + 1)) / 10
            Otheta[:, 0] = 1
            network['Otheta'] = Otheta
        else:
            Otheta = np.random.random_sample((network['Onode'], network['H1node'] + 1)) / 10
            Otheta[:, 0] = 1
            network['Otheta'] = Otheta



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
    network = C_Sub_BuildUpNetwork()
    network.build_network_main(layers=4, h1node=8, h2node=10, onode= 6)

if __name__ == '__main__':
    main()
