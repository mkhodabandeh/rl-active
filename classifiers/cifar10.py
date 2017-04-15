# from __future__ import division, print_function, absolute_import
from base_classifier import BaseClassifier 

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import numpy as np
# Data loading and preprocessing

import yaml
import tensorflow as tf

class Cifar10(BaseClassifier):
    '''
    Inspired from keras implementation of LeNet:
    https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
    
    Not a big change from this implementation.
    Only important thing to note here: there is a load_default_data method,
    and a load nondefault data method. These two methods together use preprocess_data
    to convert input to desired format, and sets training parameters.
    '''
    name = 'Cifar10'
    count = 0
    def __init__(self, config_path=None):
        if config_path:
            self.configs = yaml.load(open(config_path))
            # self.data = self.configs['data']
        else:
            self.configs = {'snapshot':'./snapshots/'}
            self.batch_size = 128
            self.epochs = 2
	self.is_annotated = set() 
        self._get_default_data()
        self._create_model()
        self.ep = 0
        
    
    def _get_default_data(self):
        # from keras.datasets import mnist
        # import tflearn.datasets.mnist as mnist
        from tflearn.datasets import cifar10
        self.num_classes = 10

        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        print 'SELF X TEST', self.x_test.shape
        self.x_train, self.y_train = shuffle(self.x_train, self.y_train)
        self.y_train = to_categorical(self.y_train, self.num_classes)
        self.y_test = to_categorical(self.y_test, self.num_classes)
        # self._preprocess_data()

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
        print('Creating Cifar10 model...')
        x_train = self.x_train[list(self.is_annotated)]
        y_train = self.y_train[list(self.is_annotated)]
        with self.sess.as_default():
            with self.graph.as_default() as g:
                self.model.fit({'input': x_train}, {'target': y_train}, n_epoch=10, validation_set=({'input': self.x_test}, {'target': self.y_test}), snapshot_step=100, show_metric=True, run_id='convnet_mnist', batch_size=96)
                # self.model.fit(x_train, y_train, n_epoch=10, shuffle=True, validation_set=(self.x_test, self.y_test),show_metric=True, batch_size=96, run_id='cifar10_cnn')
                return self._predict(x_train)

    def _predict(self, data=None):
        print('Doing Predictions...')
        with self.sess.as_default():
            with self.graph.as_default() as g:
                if data is None:
                    return np.array(self.model.predict(self.x_test))
                else:
                    return np.array(self.model.predict(data))

    def _evaluate(self, data=None):
        print('Getting Accuracy...')
        with self.sess.as_default():
            with self.graph.as_default() as g:
                if data is None:
                    self._get_default_data()
                else:
                    self._get_nondefault_data(data)
                score = self.model.evaluate(self.x_test, self.y_test)
                return score

    def _create_model(self):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.sess.as_default():
            with self.graph.as_default() as g:
                # network = input_data(shape=[None, 32, 32, 3], data_preprocessing=self.img_prep)

                img_prep = ImagePreprocessing()
                img_prep.add_featurewise_zero_center()
                img_prep.add_featurewise_stdnorm()
                self.img_prep = img_prep
                img_aug = ImageAugmentation()
                img_aug.add_random_flip_leftright()
                img_aug.add_random_rotation(max_angle=25.)
                self.img_aug = img_aug
                network = input_data(shape=[None, 32, 32, 3], name='input', data_preprocessing=self.img_prep, data_augmentation=self.img_aug)
                # network = input_data(shape=[None, 32, 32, 3] )
                network = conv_2d(network, 32, 3, activation='relu')
                network = max_pool_2d(network, 2)
                network = conv_2d(network, 64, 3, activation='relu')
                network = conv_2d(network, 64, 3, activation='relu')
                network = max_pool_2d(network, 2)
                network = fully_connected(network, 512, activation='relu')
                network = dropout(network, 0.5)
                network = fully_connected(network, 10, activation='softmax')
                network = regression(network, optimizer='adam', loss='categorical_crossentropy',name='target',learning_rate=0.001)
                # Train using classifier
                self.model = tflearn.DNN(network, tensorboard_verbose=0)
        # print('create_model',self.model.get_weights(self.w[1]))
         
    def _reset(self):
        self.sess.close()
        del self.model
        self._create_model()

    def _save_model(self):
        self.model.save(self.configs['snapshot']+'/lenet-'+self.ep)
        self.ep+=1

def test_lenet():
    lenet = LeNetTF()
    # lenet._train()
    pred_labels = lenet._predict()
    print(pred_labels.shape)
    
    # print(accuracy

if __name__ == '__main__':
    test_lenet()
