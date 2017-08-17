#!/usr/local/bin/python
# coding: GBK

__metclass__ = type

import time, progressbar, csv, math
import numpy as np
import datetime
from sqlalchemy import create_engine, MetaData, text



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

        self.TrainingSamples = []  # Matrix of Training Sample
        self.TrainingLabels = []  # Matrix of Training Labels
        self.CVSamples = []  # Matrix of CV samples
        self.CVLabels = []
        self.TestingSamples = []  # Matrix of test samples
        self.TestingLabels = []  # Marix of testing labels
        self.predicts = []  # Matrix of predicts

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

    def _save_model(self, network=None, per_log=None, mode='m'):
        if mode == 'm':
            fileName = self._output_dir + 'mode.csv'
            with open(fileName, "wb") as csvFile:
                csvWriter = csv.writer(csvFile)
                for k, v in network.iteritems():
                    csvWriter.writerow([k, v])
                csvFile.close()
        elif mode == 'p':
            fileName = self._output_dir + 'performance.csv'
            np.savetxt(fileName, per_log, delimiter=',')

    def _load_model(self, network=None):
        fileName = self._output_dir + 'Otheta_ini.csv'
        Otheta_ini = np.genfromtxt(fileName, delimiter=',', dtype=float)
        network['Otheta'] = Otheta_ini
        network['Otheta_ini'] = np.array(Otheta_ini)
        fileName = self._output_dir + 'H1theta_ini.csv'
        h1theta_ini = np.genfromtxt(fileName, delimiter=',', dtype=float)
        network['H1theta'] = h1theta_ini
        network['H1theta_ini'] = np.array(h1theta_ini)

    def load_training_set(self, model=None, training_source=None, testing_source=None, cross_validation_source=None):
        '''
        Load Data into Different DataSet
        :param model:
        :param training_source:
        :param testing_source:
        :param cross_validation_source:
        :return:
        '''
        if testing_source is None:
            testing_source = self._input_dir + 'TestingSamples.csv'
        if cross_validation_source is None:
            cross_validation_source = self._input_dir + '002310_CV_HO.csv'
        else:
            cross_validation_source = self._input_dir + cross_validation_source + '_CV_HO.csv'
        if training_source is None:
            training_source = self._input_dir + '002310_Train_HO.csv'
        else:
            training_source = self._input_dir + training_source + '_Train_HO.csv'
            # training_source = self._input_dir + 'test.csv'
        if model is None:
            model = 'Tr'

        fn = open(training_source, "r")
        trainingSamples = []
        trainingLabels = []
        for line in fn:
            xVariable = []
            line = line[:-1]  # Remove the /r/n
            vlist1 = line.split("/r")
            if vlist1[0] == "": continue  # Omit empty line
            vlist = vlist1[0].split(",")
            # if isinstance(vlist[0], str): continue  # Omit the first label line
            # Get xVariables from Training Set
            for item in vlist[:-1]:
                xVariable.append(float(item))
            trainingSamples.append(xVariable)

            # Get Lables from Training Set
            label = vlist[-1]
            #'''
            if label == 'bu':
                label_vector = [1, 0, 0, 0, 0]
            elif label == 'su':
                label_vector = [0, 1, 0, 0, 0]
            elif label == 'sd':
                label_vector = [0, 0, 0, 1, 0]
            elif label == 'bd':
                label_vector = [0, 0, 0, 0, 1]
            else:
                label_vector = [0, 0, 1, 0, 0]
            '''
            if label == '1':
                label_vector = [1, 0]
            else:
                label_vector = [0, 1]
            '''
            trainingLabels.append(label_vector)

        print "Loaded %s Training Samples" % len(self.TrainingSamples)
        self.TrainingSamples = np.array(trainingSamples)
        # bias = np.ones((self.TrainingSamples.shape[0], 1))
        # self.TrainingSamples = np.c_[bias, self.TrainingSamples]
        self.TrainingLabels = np.array(trainingLabels)
        fn.close()

        if model == 'Testing':
            fn = open(testing_source, 'r')
            testingSamples = []
            testingLabels = []
            for line in fn:
                xVariable = []

                line = line[:-1]  # Remove the /r/n
                vlist1 = line.split("/r")
                if vlist1[0] == "": continue
                vlist = vlist1[0].split(",")
                # Get xVariables from Testing Set
                for item in vlist[:-1]:
                    xVariable.append(float(item))
                testingSamples.append(xVariable)

                # Get Lables from Testing Set
                label = vlist[-1]
                if label == 'bu':
                    label_vector = [1, 0, 0, 0, 0]
                elif label == 'su':
                    label_vector = [0, 1, 0, 0, 0]
                elif label == 'sd':
                    label_vector = [0, 0, 0, 1, 0]
                elif label == 'bd':
                    label_vector = [0, 0, 0, 0, 1]
                else:
                    label_vector = [0, 0, 1, 0, 0]
                testingLabels.append(label_vector)
            print "Loaded %s Testing Samples" % len(self.TestingSamples)
            self.TestingSamples = np.array(testingSamples)
            # bias = np.ones((self.TestingSamples.shape[0], 1))
            # self.TestingSamples = np.c_[bias, self.TestingSamples]
            self.TestingLabels = np.array(testingLabels)
            fn.close()
        elif model == 'CV':
            fn = open(cross_validation_source, 'r')
            cvSamples = []
            cvLabels = []
            for line in fn:
                xVariable = []
                line = line[:-1]  # Remove the /r/n
                vlist1 = line.split("/r")
                if vlist1[0] == "": continue
                vlist = vlist1[0].split(",")
                # Get xVariables from CV Set
                for item in vlist[:-1]:
                    xVariable.append(float(item))
                cvSamples.append(xVariable)
                # Get Lables from CV Set
                label = vlist[-1]
                if label == 'bu':
                    label_vector = [1, 0, 0, 0, 0]
                elif label == 'su':
                    label_vector = [0, 1, 0, 0, 0]
                elif label == 'sd':
                    label_vector = [0, 0, 0, 1, 0]
                elif label == 'bd':
                    label_vector = [0, 0, 0, 0, 1]
                else:
                    label_vector = [0, 0, 1, 0, 0]
                cvLabels.append(label_vector)
            print "Loaded %s CV Samples" % len(self.CVSamples)
            self.CVSamples = np.array(cvSamples)
            # bias = np.ones((self.CVSamples.shape[0], 1))
            # self.CVSamples = np.c_[bias, self.CVSamples]
            self.CVLabels = np.array(cvLabels)
            fn.close()
        else:
            pass
        return

    def _add_bias(self, input_set):
        '''
        Add bias term for layer input
        :return:
        '''
        bias = np.ones((input_set.shape[0], 1))*1.0
        biased_set = np.c_[bias, input_set]
        return biased_set

class C_Sub_BuildUpNetwork(C_BPNetwork):
    def __init__(self):
        C_BPNetwork.__init__(self)
        self.network = {
            'layer': 0,
            'inode': 0,
            'h1node': 0,
            'h2node': 0,
            'onode': 0,
            'H1ParameterNumbers': 0,
            'H1theta_ini': [],  # Hidden Layer 1 initial Parameters
            'H1theta': [],  # Hidden Layer 1 t step Parameters
            'H1theta_t1': [],  # Hidden Layer 1 t-1 step parameters
            'H1output': [],  # Hidden Layer 1 output
            'H1error': [],  # Hidden Layer 1 errors
            'H1delta': [],  # Hidden Layer 1 delta
            'H1delta_t1': [],  # Hidden Layer 1 t-1 step delta
            'H1bias': [],  # Hidden Layer 1 bias
            'H2ParameterNumbers': 0,
            'H2theta_ini': [],  # Hidden Layer 2 initial Parameters
            'H2theta': [],  # Hidden Layer 2 t step Parameters
            'H2theta_t1': [],  # Hidden Layer 2 t-1 step parameters
            'H2output': [],  # Hidden Layer 2 output
            'H2error': [],  # Hidden Layer 2 error
            'H2delta': [],  # Hidden Layer 2 delta
            'H2delta_t1': [],  # Hidden Layer 1 t-1 step delta
            'H2bias': [],  # Hidden Layer 2 bias
            'OParameterNumbers': 0,
            'Otheta_ini': [],  # Output layer initial Parameters
            'Otheta': [],  # Output layer t step Parameters
            'Otheta_t1': [],  # Output Layer 1 t-1 step parameters
            'Ooutput': [],  # Ouput Layer 2 output
            'Oerror': [],  # Output Layer 2 error
            'Odelta': [],  # Output Layer Delta
            'Odelta_t1': [],  # Output Layer t-1 step delta
            'Obias': [],  # Output layer Bias
            'BatchMode': True,  # Switch of Batch Mode
            'BCost': 0.0,  # batch Cost for whole Training Set
            'SCost': 0.0,  # Single Cost for each Training sample
            'CT': 0.0,  # Cost Threshold, the traget level of cost
            'CRound': 0,  # Current Training Rounds
            'MRound': 0,  # Max Training Rounds
            'lambda': 0.0,  # Regularization Parameter
            'mu': 0.0,  # Learning Rate
            'epsilon': 0.0,  # Step length of Calculating the Derivative of parameter theta
            'momentum': False,  # Switch of Momentum model
            'alpha': 0.0,  # Parameter of momentum term
            'epslon':0.0 # Parameter of Random initialization
        }

    def build_network_main(self, layers=None, inode=None, h1node=None, h2node=None, onode=None, batch_mode=True,
                           momentum=False, alpha=0.3, MRound=10, mu=0.7, regular_term=3.0, epslon=0.2):
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

        self.network['layer'] = layers,
        self.network['inode'] = inode,
        self.network['h1node'] = h1node,
        self.network['h2node'] = h2node,
        self.network['onode'] = onode,
        self.network['BatchMode'] = batch_mode,  # Switch of Batch Mode

        # Random initiliaztion Parameter. Make sure the initial parameter is in [-epslon , epslon]
        self.epslon = epslon
        # Initialize the variation length of Theta, normally use 2epsilon to calculate the two sides derivative.
        self.network['epslon'] = epslon
        # Set the allowed Max Rounds of Training:
        self.network['MRound'] = MRound
        # Initialize the Regularization parameter, regularization is to balance the current delta and last theta
        self.network['lambda'] = regular_term
        # Initialize the Learning Rate
        self.network['mu'] = mu
        # Initialize the Momentum model
        if momentum:
            self.network['momentum'] = True
            self.network['alpha'] = alpha
        # Initialize the Parameters for each node.
        # The first columns of each set of parameter are 1 corresponding to the Bias.
        self._initialize_parameters(self.network)
        # Save Mode
        self._save_model(network=self.network, mode='m')
        # print self.network

    def _initialize_parameters(self, network):
        '''
        This fucntion is to initialize origin prameters for the network.
        The parameters are randoms selected number between 0 and 1.
        The parameters should be small enough
        :param network:
        :return:
        '''
        # H1theta = np.random.random_sample((network['h1node'][0], network['inode'][0] + 1))
        # H1theta = np.random.random_sample((network['h1node'][0], network['inode'][0] + 1))*(2*network['epslon'])-network['epslon']
        H1theta = np.random.uniform(low=0.1, high=2.0, size=(network['h1node'][0], network['inode'][0])) * (
        2 * network['epslon']) - \
                  network['epslon']
        network['H1theta']=H1theta
        network['H1theta_ini'] = np.array(H1theta)
        network['H1bias'] = np.ones((1, network['h1node'][0]))
        # H1theta_2 = np.repeat(np.random.random_sample((1, network['inode'][0] + 1)),network['h1node'][0], axis=0)
        #network['H1theta'] = H1theta_2
        if network['h2node'][0] != 0:
            # H2theta = np.random.random_sample((network['h2node'][0], network['h1node'][0] + 1))*(2*network['epslon'])-network['epslon']
            H2theta = np.random.uniform(low=0.1, high=2.0, size=(network['h2node'][0], network['h1node'][0])) * (
                2 * network['epslon']) - network['epslon']
            network['H2theta'] = H2theta
            network['H2theta_ini'] = np.array(H2theta)
            network['H2bias'] = np.ones((1, network['h2node'][0]))
            # H2theta_2 = np.repeat(np.random.random_sample((1, network['h1node'][0] + 1)), network['h2node'][0], axis=0)
            # network['H2theta'] = H2theta_2
            # Otheta = np.random.random_sample((network['onode'][0], network['h2node'][0] + 1))*(2*network['epslon'])-network['epslon']
            Otheta = np.random.uniform(low=0.1, high=2.0, size=(network['onode'][0], network['h2node'][0])) * (
                2 * network['epslon']) - network['epslon']
            network['Otheta'] = Otheta
            network['Otheta_ini'] = np.array(Otheta)
            network['Obias'] = np.ones((1, network['onode'][0]))
            # Otheta_2 = np.repeat(np.random.random_sample((1, network['h2node'][0] + 1)), network['onode'][0], axis=0)
            #network['Otheta'] = Otheta_2
        else:
            # Otheta = np.random.random_sample((network['onode'][0], network['h1node'][0] + 1))*(2*network['epslon'])-network['epslon']
            Otheta = np.random.uniform(low=0.1, high=2.0, size=(network['onode'][0], network['h1node'][0])) * (
                2 * network['epslon']) - network['epslon']
            network['Otheta'] = Otheta
            network['Otheta_ini'] = np.array(Otheta)
            network['Obias'] = np.ones((1, network['onode'][0]))
            # Otheta_2 = np.repeat(np.random.random_sample((1, network['h1node'][0] + 1)), network['onode'][0], axis=0)
            #network['Otheta'] = Otheta_2

class C_Sub_TrainNetwork(C_BPNetwork):
    def __init__(self):
        C_BPNetwork.__init__(self)

    def Train_Main(self, network, mode=None, training_source=None, CV_source=None):
        if mode is None:
            mode = 'B'  # Batch Train or Singal Train, mode = S is online training

        # load Training Data
        self.load_training_set(model='CV', training_source=training_source, cross_validation_source=CV_source)

        # Total Cost Monitor
        total_cost = np.ones((network['MRound'], 5))
        # total_cost=[[1,1,1,1,1]]*network['MRound']

        # Training Process control
        '''
        control1 = raw_input("Please input: Y or N")
        if control1 == 'N' or control1 == 'n':
            return
        else:
            print "Starting Training Process. \n"
        '''
        # Training loop
        i = 0
        while i < network['MRound']:
            if mode == 'B':
                print "Round %s ---------------------------------------------------------" % network['CRound']
                # Forward Calculation
                loop_total_cost = self._forward_calculation(network, mode, source='Train')
                network['BCost'] = loop_total_cost
                total_cost = self._training_control(network, total_cost, loop_total_cost, i)
                #print "For Round # %s, the total cost is %s \n" % (i, loop_total_cost)
                # Backword Calculation
                network = self._backward_calculation(network, mode)
                print network['Otheta'][:3, :]
            else:  # SGD
                network = self.SGD_Training(network)
                cost = self._cal_sig_cost(network)
                print "For round %s, cost is %s" % (i, cost)
            # Add 1 loop
            i += 1
            network['CRound'] = i
        self._save_model(per_log=total_cost, mode='p')
        self._save_model(network=network, mode='m')

        return network

    def SGD_Training(self, network):
        h1theta = network['H1theta']
        otheta = network['Otheta']
        mu = network['mu']
        regular = network['lambda']
        for idx in range(0, len(self.TrainingSamples)):
            # for idx in range(0,1):
            # Froward Calculation
            sample = np.array(self.TrainingSamples[idx])
            h1ac = self._cal_activation((np.dot(h1theta, sample) + network['H1bias']))
            oac = self._cal_activation((np.dot(h1ac, np.transpose(otheta)) + network['Obias']))

            # BackPropogation
            odev = oac * (1 - oac)
            h1dev = h1ac * (1 - h1ac)
            oerror = (np.array(self.TrainingLabels[idx]) - oac) * odev
            odelta = np.dot(np.transpose(oerror), h1ac)
            network['Otheta'] += odelta * network['mu']
            network['Obias'] += np.sum(oerror, axis=0, keepdims=True) * network['mu']
            h1error = np.dot(oerror, otheta) * h1dev
            sample = sample.reshape((-1, 1)).reshape((1, -1))
            h1delta = np.dot(np.transpose(h1error), sample)
            network['H1theta'] += h1delta * network['mu']
            network['H1bias'] += np.sum(h1error, axis=0, keepdims=True) * network['mu']

        return network


    def Cross_Validation(self, network):
        Ooutput = self._forward_calculation(network, mode='P', source='CV')
        max_index = np.argmax(Ooutput, axis=1)
        prediction = np.zeros(Ooutput.shape)
        i = 0
        print max_index.shape
        while i < max_index.shape[0]:
            index = max_index[i]
            prediction[i][index] = 1
            i+= 1
        print Ooutput
        #print prediction
        #print self.CVLabels
        #print (prediction - self.CVLabels)

    def Cross_Validation_SGD(self, network):
        h1theta = network['H1theta']
        otheta = network['Otheta']
        labels = np.array(self.CVLabels)
        Ooutput = np.zeros(labels.shape)
        for idx in range(0, len(self.CVSamples)):
            # for idx in range(0,1):
            # Froward Calculation
            sample = np.array(self.CVSamples[idx])
            label = np.array(self.CVLabels[idx])
            h1ac = self._cal_activation((np.dot(h1theta, sample) + network['H1bias']))
            oac = self._cal_activation((np.dot(h1ac, np.transpose(otheta)) + network['Obias']))
            Ooutput[idx, :] = Ooutput[idx, :] + oac

        max_index = np.argmax(Ooutput, axis=1)
        prediction = np.zeros(Ooutput.shape)
        i = 0
        while i < max_index.shape[0]:
            index = max_index[i]
            prediction[i][index] = 1
            i += 1
        # print "Prediction is \n %s"%prediction
        # print "Labels is \n %s"%self.CVLabels
        # print "Difference is \n %s" % (prediction-labels)
        print "Accuracy is %s" % round((np.count_nonzero(prediction - labels) /2),2)


    def _training_control(self, network, total_cost, loop_total_cost, round):
        '''
        Training control is a function to speed up or slowing down the training process by monitoring the
        total cost performance and tunning the learning rates.
        :return:
        '''
        total_cost[round][0] = int(round)
        total_cost[round][1] = loop_total_cost
        total_cost[round][4] = network['mu']
        print "For Round # %s, the total cost is %s \n" % (round, loop_total_cost)

        if round > 0:
            round_diff = (total_cost[round - 1][1] - total_cost[round][1])
            total_cost[round][2] = round_diff
            if round_diff / total_cost[round - 1][2] > 0:
                total_cost[round][3] = math.log(round_diff / total_cost[round - 1][2])
                # Dynamic learning rates
                if total_cost[round][3] > -0.03:
                    # increase learning rates
                    network['mu'] += 0.01
                elif total_cost[round][3] < -0.05:
                    network['mu'] -= 0.01
            else:
                total_cost[round][3] = -100
                # decrease learning rates
                # network['mu'] -= 0.03

        return total_cost

    def _forward_calculation(self, network, mode, source=None):
        '''

        :param network:
        :param mode: # Batch Training or Single Training
        :return:
        '''
        # Calculate output value of Hidden Layer 1
        if source=='Train':
            input_set = self.TrainingSamples
        elif source == 'CV':
            input_set = self.CVSamples
        elif source == 'Testing':
            input_set = self.TestingSamples
        else:
            input_set = source

        #Calculate Hidden Layer 1
        theta_set = np.transpose(network['H1theta'])
        h1output = self._cal_layer_output(theta_set=theta_set, input_set=input_set)
        network['H1output'] = h1output

        print network['layer']
        # Calculate Hidden Layer 2
        if network['layer'][0] == 4:
            h2input = self._add_bias(h1output)
            theta_set = np.transpose(network['H2theta'])
            h2output = self._cal_layer_output(theta_set=theta_set, input_set=h2input)
            network['H2output'] = h2output
            oinput = self._add_bias(h2output)
            print "Calculating layer 4"
        else:
            oinput = self._add_bias(h1output)

        # Calculate output layer
        otheta_set = np.transpose(network['Otheta'])
        ooutput = self._cal_layer_output(theta_set=otheta_set, input_set=oinput)
        network['Ooutput'] = ooutput

        # Calculate Forward Propagation Cost
        if mode == 'B': # Batch calculation mode
            total_cost = self._cal_cost_function(ooutput, mode, network)
            return total_cost
        elif mode == 'P': # Prediction Mode
            return network['Ooutput']

    def _cal_layer_output(self, theta_set, input_set):
        '''
        This function is to calculate the nodes output of each layer.
        :param theta_set: Is a np.array of the parameters of a layer
        :param layer_input: Is a np.array of the inputs for a layer
        :return: output value
        '''
        # print "Before activation"
        #print np.dot(input_set, theta_set)
        output = self._cal_activation(np.dot(input_set, theta_set))
        return output

    def _cal_activation(self, in_value):
        # activated_value = 1.0 / (1.0 + np.exp(in_value))
        activated_value = 0.5 * (1 + np.tanh(0.5 * in_value))
        return activated_value

    def _cal_cost_function(self, ooutput, mode, network):
        '''
        Cost calculation of Neural Network
        :param ooutput:
        :param mode: Batch mode or Single Mode
        :return:
        '''
        cost = 0
        if mode == 'B':  # Batch Calculation
            labels = self.TrainingLabels
            count = labels.shape[0]
            # Individal Cost
            np.log(ooutput)
            individal_costs = np.dot(labels, np.transpose(np.log(ooutput))) + np.dot((1 - labels),
                                                                                     np.transpose(
                                                                                         np.log(1.0 - ooutput)))

            # Total Cost Calculation
            total_cost = np.sum(individal_costs) / (-1 * count)

            # Regularization term calculation
            regularization_term = 0
            h1theta = np.power(np.array(network['H1theta'])[:, 1:], 2)
            regularization_term += np.sum(h1theta)

            if network['layer'][0] == 4:
                h2theta = np.power(np.array(network['H2theta'])[:, 1:], 2)
                regularization_term += np.sum(h2theta)
            otheta = np.power(np.array(network['Otheta'])[:, 1:], 2)
            regularization_term += np.sum(otheta)
            regularization_term = (regularization_term * network['lambda']) / (2 * count)

            # Final Cost
            total_cost = total_cost + regularization_term
            return total_cost
        else:
            pass

        return cost

    def _cal_sig_cost(self, network):
        h1theta = network['H1theta']
        otheta = network['Otheta']
        cost = 0
        for idx in range(0, len(self.TrainingSamples)):
            # for idx in range(0,1):
            # Froward Calculation
            sample = np.array(self.TrainingSamples[idx])
            label = np.array(self.TrainingLabels[idx])
            h1ac = self._cal_activation((np.dot(h1theta, sample) + network['H1bias']))
            oac = self._cal_activation((np.dot(h1ac, np.transpose(otheta)) + network['Obias']))
            cost += np.sum(np.power((label - oac), 2)) / 2
        return cost

    def _backward_calculation(self, network, mode):
        network = self._cal_error(network, mode)
        network = self._cal_delta_main(network, mode)
        network = self._update_theta(network)
        return network

    def _cal_error(self, network, mode):
        '''
        Errors calculation for each layer in Back propagation
        :param network:
        :return:
        '''
        # Calculate output level errors
        ooutput = network['Ooutput']
        labels = self.TrainingLabels
        output_error = ooutput - labels
        network['Oerror'] = output_error
        #print "Ooutlevel Errors: \n %s"%output_error


        # Calculate H2 layer errors
        if network['layer'][0] == 4:
            error_upper = network['Oerror']
            Otheta = network['Otheta']
            h2output = network['H2output']
            error_Otheta = np.dot(error_upper, Otheta)[:, 1:]
            h2error = error_Otheta * h2output * (1 - h2output)
            network['H2error'] = h2error
            error_upper = network['H2error']
            theta_upper = network['H2theta']
        else:
            error_upper = network['Oerror']
            theta_upper = network['Otheta']

        # Calculate H1 layer errors
        h1output = network['H1output']
        error_Otheta = np.dot(error_upper, theta_upper)[:, 1:]
        h1error = error_Otheta * (h1output * (1 - h1output))
        network['H1error'] = h1error

        return network

    def _cal_delta_main(self, network, mode):

        # Calculate the Delta for hidden layer 2
        if network['layer'][0] == 4:
            delta_sum_o = self._cal_delta_sum(network, 'o', 'h2')
            delta_sum_h2 = self._cal_delta_sum(network, 'h2', 'h1')
            if network['momentum']:
                self._delta_update_momentum(network, delta_sum_o, 'Odelta')
                self._delta_update_momentum(network, delta_sum_h2, 'H2delta')
            else:
                network['Odelta'] = self._cal_delta_learning_rate(network, delta_sum_o)
                network['H2delta'] = self._cal_delta_learning_rate(network, delta_sum_h2)
        # Calculate the Delta for hidden layer 1
        else:
            delta_sum_o = self._cal_delta_sum(network, 'o', 'h1')
            if network['momentum']:
                self._delta_update_momentum(network, delta_sum_o, 'Odelta')
            else:
                network['Odelta'] = self._cal_delta_learning_rate(network, delta_sum_o)
                #print "Output level delta is \n %s" %self._cal_delta_learning_rate(network, delta_sum_o)

        # Calculate the Delta for input layer
        delta_sum_h1 = self._cal_delta_sum(network, 'h1', 'i')
        if network['momentum']:
            self._delta_update_momentum(network, delta_sum_h1, 'H1delta')
        else:
            network['H1delta'] = self._cal_delta_learning_rate(network, delta_sum_h1)
        #print "H1 level delta is \n %s" % self._cal_delta_learning_rate(network, delta_sum_h1)
        return network

    def _delta_update_momentum(self, network, delta_sum, delta_layer):
        delta_t1 = delta_layer + '_t1'
        if network[delta_layer] != []:
            network[delta_t1] = np.array(network[delta_layer])
            network[delta_layer] = self._cal_delta_learning_rate(network, delta_sum) + network['alpha'] * network[
                delta_t1]
        else:
            network[delta_layer] = self._cal_delta_learning_rate(network, delta_sum)
            network[delta_t1] = np.zeros(network[delta_layer].shape)

    def _cal_delta_sum(self, network, higher_layer, lower_layer):
        '''
        Calculation of Summation of Delta for each unit
        :param network:
        :param higher_layer:
        :param lower_layer:
        :return:
        '''
        h_error = higher_layer.capitalize() + 'error'
        h_node = higher_layer + 'node'
        l_output = lower_layer.capitalize() + 'output'
        l_node = lower_layer + 'node'

        if lower_layer == 'i':
            # lower_output = self.TrainingSamples[:, 1:]
            lower_output = self.TrainingSamples
        else:
            # lower_output = network[l_output]
            lower_output = self._add_bias(network[l_output])

        higher_error = network[h_error]
        delta_sum = np.zeros((network[h_node][0], network[l_node][0] + 1))
        for i in range(lower_output.shape[0]):
            #'''
            delta = np.ones((network[h_node][0], network[l_node][0] + 1))
            output_trans = lower_output[i]
            delta = delta * output_trans
            error_trans = higher_error[i]
            delta = delta.T * error_trans
            delta_sum = delta_sum + delta.T
            '''
            delta = np.ones((1, network[l_node][0] + 1))
            error_trans = higher_error[i]
            delta = np.dot(error_trans.reshape((-1,1)), delta)
            output_trans = lower_output[i]
            delta = delta * output_trans
            delta_sum = delta_sum + delta
            '''
        #print "%s level delta summation is %s"%(h_error, delta_sum)
        return delta_sum

    def _cal_delta_learning_rate(self, network, delta_sum):
        '''
        What is the learning rate, where to use the learning rate need to be considered.
        :param network:
        :param delta_sum:
        :return:
        '''
        sample_count = self.TrainingSamples.shape[0]
        learning_rate = network['mu']
        delta = (learning_rate * delta_sum) / sample_count
        #delta = (learning_rate * delta_sum)
        return delta

    def _update_theta(self, network):
        regularizaion_term = network['lambda']
        # Update H1 level parameters
        network['H1theta_t1'] = np.array(network['H1theta'])
        step1 = network['H1delta'][:, 1:] + regularizaion_term * network['H1theta_t1'][:, 1:]
        last_step = network['H1theta_t1'][:, 1:]
        network['H1theta'][:, 1:] = last_step - step1
        # update Bias
        #wrong
        network['H1theta'][:, :1] = network['H1theta_t1'][:, :1] - network['H1delta'][:, :1]

        # Update H2 level parameters
        if network['layer'][0] == 4:
            network['H2theta_t1'] = np.array(network['H2theta'])
            network['H2theta'][:, 1:] = network['H2theta_t1'][:, 1:] - (
            network['H2delta'][:, 1:] + regularizaion_term * network['H2theta_t1'][:, 1:])
            network['H2theta'][:, :1] = network['H2theta_t1'][:, :1] - network['H2delta'][:, :1]

        # Update Output level parameters
        network['Otheta_t1'] = np.array(network['Otheta'])
        network['Otheta'][:, 1:] = network['Otheta_t1'][:, 1:] - (
        network['Odelta'][:, 1:] + regularizaion_term * network['Otheta_t1'][:, 1:])
        #Update Bias
        network['Otheta'][:, :1] = network['Otheta_t1'][:, :1] - network['Odelta'][:, :1]
        return network

class C_Sub_NetworkPrediction(C_BPNetwork):
    def __init__(self):
        C_BPNetwork.__init__(self)


    def predict_main(self, mode=None):
        if mode is None:
            mode = 'CV'
        else:
            mode = 'Te'
        # load CV Data
        self.load_training_set(model='CV')


        self._load_model()
        pass


def main():
    network = C_Sub_BuildUpNetwork()
    network.build_network_main(layers=3, inode=20, h1node=10, h2node=0, onode=5, MRound=1000, regular_term=1, mu=0.1,
                               momentum=False, alpha=0.1, epslon=0.2)
    # '''
    # network.build_network_main(layers=3, inode=5, h1node=3, h2node=0, onode=2, MRound=2,regular_term=1, mu=0.1,
    #                          momentum=True, alpha=0.5, epslon=0.9)
    network._load_model(network.network)
    train = C_Sub_TrainNetwork()
    trained_network = train.Train_Main(network.network, training_source='002310', CV_source='002310', mode='s')
    # trained_network = train.Train_Main(network.network, training_source='002310', CV_source='002310')
    # train.Cross_Validation(trained_network)
    train.Cross_Validation_SGD(trained_network)
    #'''

if __name__ == '__main__':
    main()
