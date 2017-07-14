# from __future__ import division, print_function, absolute_import
import os

#see: https://github.com/tensorflow/tensorflow/issues/566

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
from base_classifier import BaseClassifier 

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import numpy as np
# Data loading and preprocessing
import subprocess
current_user = subprocess.check_output(['whoami']).strip()
# current_user = result.stdout

import yaml

print 'Lenet file loaded'
# classifier_args = config['classifiers']
# locals().update(rl_args)
class LeNetTF(BaseClassifier):
    '''
    Inspired from keras implementation of LeNet:
    https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
    
    Not a big change from this implementation.
    Only important thing to note here: there is a load_default_data method,
    and a load nondefault data method. These two methods together use preprocess_data
    to convert input to desired format, and sets training parameters.
    '''
    name = 'LeNet_TF'
    count = 0
    def __init__(self, config_path=None, device=None, num_data=200):
        self.device = device

        # tf_config.allow_soft_placement = True
        
        if 'gpu' in device:
            tf_config = tf.ConfigProto(device_count={'GPU':1})
            tf_config.allow_soft_placement = True 
            tf_config.gpu_options.allow_growth = True
        else:
            tf_config = tf.ConfigProto(device_count={'CPU':8}, 
                                        inter_op_parallelism_threads=8,
                                        intra_op_parallelism_threads=1)
        # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.6
        # tf_config.log_device_placement = True
        
        self.tf_config=tf_config
        self.graph = tf.Graph()
        self.sess = tf.Session(config=tf_config, graph=self.graph)
        if config_path:
# config = yaml.load(open('config.yml'))
            #self.configs = yaml.load(open(config_path))
            self.num_data = num_data#self.configs['SHARED']['NUM_DATA']
            # self.num_data = self.configs['SHARED']['NUM_DATA']
            # self.data = self.configs['data']
        else:
            self.configs = {'snapshot':'./snapshots/'}
        self.batch_size = 128 
        self.epochs = 100 
	self.is_annotated = set() 
        self._get_default_data()
        self._create_model()
        self.ep = 0
        
    def _preprocess_data(self):
        #only a portion of the data is used for training the RL agent
        self.x_train = self.x_train[:self.num_data, ...]
        self.x_train = self.x_train.reshape([-1, 28, 28, 1])
        self.x_test = self.x_test.reshape([-1, 28, 28, 1])
    
    def _get_default_data(self):
        # from keras.datasets import mnist
        import tflearn.datasets.mnist as mnist
        self.num_classes = 10

        self.x_train, self.y_train, self.x_test, self.y_test = mnist.load_data(one_hot=True)
        self.default_loaded = True
        self._preprocess_data()

        print('x_train shape:', self.x_train.shape)
        print(self.x_train.shape[0], 'train samples')
        print(self.x_test.shape[0], 'test_samples')

    def _get_class_n(self):
        """
        This function gives number of classes and number of training samples
        """
        return  self.num_classes

    def _get_train_n(self):
        """
        This function gives number of classes and number of training samples
        """
        return self.x_train.shape[0]



    def _train(self, data=None):
        print('[lenet_tflearn]Creating MNIST model...')
        if self.is_annotated:
            x_train = self.x_train[list(self.is_annotated)]
            y_train = self.y_train[list(self.is_annotated)]
        else:
            x_train = self.x_train
            y_train = self.y_train
        # print x_train.shape
        # print y_train.shape
        #with self.sess.as_default():
        try:
            with self.graph.as_default() as g:
                print '[lenet_tflearn.py::_train] size of input', x_train.shape 
                with tf.device(self.device):
                    self.model.fit({'lenet_input': x_train}, {'target': y_train}, n_epoch=self.epochs, validation_set=({'lenet_input': self.x_test}, {'target': self.y_test}), batch_size=self.batch_size, snapshot_step=128, show_metric=True, run_id='convnet_mnist')
        except Exception as e:
            print '[lenet_tflearn]EXCEPTION', x_train.shape, y_train.shape
            raise e
            # exit()
        # return self._predict(self.x_train)
        return self._predict(self.x_train)

    def _predict(self, data=None):
        print('[lenet_tflearn]Doing Predictions...')
        with self.sess.as_default():
            with self.graph.as_default() as g:
                with tf.device(self.device):
                    if data is None:
                        data = self.x_train

                    arr = np.zeros((self.x_train.shape[0], self.num_classes))
                    # print arr.shape
                    batch_size = 128
                    for i in xrange(self.x_train.shape[0]/batch_size):
                        # arr[i*batch_size:(i+1)*batch_size,...] =  
                        temp = np.array(self.model.predict({'lenet_input': self.x_train[i*batch_size:(i+1)*batch_size, ...]}))
                        # print temp.shape
                        arr[i*batch_size:(i+1)*batch_size, ...] = temp
                        # print i*batch_size, (i+1)*batch_size
                        # print temp.shape
                        # print temp[0:2, ...]
                        # pdb.set_trace()
                    # print (i+1)*batch_size, self.x_train.shape[0]
                    i = self.x_train.shape[0]/batch_size
                    if (i)*batch_size != self.x_train.shape[0]:
                        # print 'hi'
                        temp = np.array(self.model.predict({'lenet_input': self.x_train[(i)*batch_size:, ...]}))
                        arr[(i)*batch_size:, ...] = temp

                    # return np.array(self.model.predict({'lenet_input': self.x_train}))
                    return arr

    def _evaluate(self, data=None):
        print('[lenet_tflearn]Getting Accuracy...')
        with self.sess.as_default():
            with self.graph.as_default() as g:
                if data is None:
                    if not self.default_loaded: 
                        self._get_default_data()
                else:
                    self._get_nondefault_data(data)
                with tf.device(self.device):
                    print '[lenet_tflearn] evaluating'
                    score = self.model.evaluate(self.x_test, self.y_test, batch_size=128)
                    return score

    def _create_model(self):
        # self.graph = tf.Graph()
        # self.sess = tf.Session(graph=self.graph, config=self.tf_config)
        print '[lenet_tflearn]: DEVICE IS', self.device
        with tf.device(self.device):
            with self.graph.as_default() as g:
                network = input_data(shape=[None, 28, 28, 1], name='lenet_input')
                network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
                network = max_pool_2d(network, 2)
                network = local_response_normalization(network)
                network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
                network = max_pool_2d(network, 2)
                network = local_response_normalization(network)
                network = fully_connected(network, 128, activation='tanh', name='fc1')
                # self.w = tflearn.variables.get_layer_variables_by_name('fc1')
                network = dropout(network, 0.8 )
                network = fully_connected(network, 256, activation='tanh')
                network = dropout(network, 0.8)
                network = fully_connected(network, 10, activation='softmax')
                network = regression(network, optimizer='adam', learning_rate=0.01,
                                                loss='categorical_crossentropy', name='target')
                # try:
                    # init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                    # self.sess.run(tf.variables_initializer(tf.get_collection_ref('is_training')))
                # except:
                    # init = tf.initialize_all_variables()
                # self.sess.run(init)
        # self.sess.run(tf.global_variables_initializer())
        with tf.device(self.device):
            with self.graph.as_default() as g:
                self.model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='/tmp/'+current_user+'/tflearn_log', session=self.sess)
                # self.graph.finalize()
                try:
                    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                    self.sess.run(tf.variables_initializer(tf.get_collection_ref('is_training')))
                except:
                    init = tf.initialize_all_variables()
                self.sess.run(init)
        # print('create_model',self.model.get_weights(self.w[1]))
         
    def _reset(self):
        tf.reset_default_graph()
        self.sess.close()
        self.graph = tf.Graph()
        self.sess = tf.Session(config=self.tf_config, graph=self.graph)
        
        del self.model
        # import ipdb
        # ipdb.set_trace()
        self._create_model()

    def _save_model(self):
        self.model.save(self.configs['snapshot']+'/lenet-'+self.ep)
        self.ep+=1

def test_lenet(num_data, init_n, sample_n, gpu_n):
    lenet = LeNetTF(config_path='./config.yml',device='/gpu:{0}'.format(gpu_n}, num_data=num_data)
    # print 'HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH'*10
    accs = []
    lenet.epochs = 20
    import random
    for i in xrange(1):
        # lenet.set_annotations(range(0,400))
        lenet.set_set_annotations(random.sample(range(0,num_data), sample_n))
        lenet.train()
        accs.append(lenet.evaluate())
        lenet._reset()
    print'numdata_{0} init_{1} sample_{2} epochs_{3}'.format(lenet.num_data,init_n, sample_n, lenet.epochs)  
    print accs
    accs2 = []
    mean =np.mean(np.array(accs)) 
    std =np.std(np.array(accs))
    print 'mean', mean, 'std', std
    output = open('./logs/numdata_{0}_init_{1}_sample_{2}_epochs_{3}.txt'.format(lenet.num_data,init_n, sample_n, lenet.epochs), 'w')
    output_arr = [accs, mean, std] 
    output.write(str(output_arr))
    output.close()

if __name__ == '__main__':
    import sys
    num_data = int(sys.argv[1])
    init_n = int(sys.argv[2])
    sample_n= int(sys.argv[3])
    gpu_n = int(sys.argv[4])
    test_lenet(num_data, init_n, sample_n, gpu_n)
