from __future__ import print_function
import sys

def eprint(*args, **kwargs):
        print(*args, file=sys.stderr, **kwargs)

from keras.models import *
from keras.layers import *
from keras import backend as K
import tensorflow as tf
STATE_SIZE = 128 
NUM_CLASSES =  10 
NUM_DATA = 5 
def _build_dynamic_model(self, is_annotated):

        # create the network
        not_annotated = set(range(NUM_DATA)).difference(is_annotated)
        inputs = []
        for i in xrange(NUM_DATA):
            # inputs.append(tf.placeholder(tf.float32, shape=(None, NUM_CLASSES)))
            inputs.append(Input(shape=(NUM_CLASSES,)))
        state_outputs = []
        for a_i in is_annotated:
            phi_s_out =self.phi_s_model(inputs[a_i]) 
            phi_s_out = Reshape((1,STATE_SIZE))(phi_s_out)
            state_outputs.append(phi_s_out)
                    
        s_concat = concatenate(state_outputs, axis=1)
        phi_state = GlobalAveragePooling1D()(s_concat)
        pi_outputs = []
        for a_i in not_annotated:
            pi_i = self.pi_model([inputs[a_i], phi_state])#PI(a_i|phi(s))
            pi_outputs.append(pi_i)

        termination_action = self.termination_action_model(phi_state)
        pi_outputs.append(termination_action)

        action_concat = concatenate(pi_outputs, axis=1)
        out_actions = Activation('softmax')(action_concat)
        out_value = self.v_model(phi_state)
        final_model = Model(inputs=inputs, outputs=[out_actions, out_value])
        final_model._make_predict_function()

        return final_model

def _build_termination_action_model():
        # Builds the state graph
        # Input is an image class probabilities
        # output is the state vector with size STATE_SIZE
        l_input = Input( shape=(STATE_SIZE,) )
        l_dense1 = Dense(64, activation='elu')(l_input)
        out_state = Dense(1, activation='elu')(l_dense1)
        model = Model(inputs=l_input,outputs=out_state)
        # model._make_predict_function()	# have to initialize before threading
        return model
def _build_phi_s_model():
        # Builds the state graph
        # Input is an image class probabilities
        # output is the state vector with size STATE_SIZE
        l_input = Input(shape=(NUM_CLASSES,) )
        l_dense = Dense(128, activation='elu')(l_input)
        l_dense1 = Dense(64, activation='elu')(l_dense)
        out_state = Dense(STATE_SIZE, activation='elu')(l_dense1)
        model = Model(inputs=l_input,outputs=out_state)
        # model._make_predict_function()	# have to initialize before threading
        return model

def _build_v_model():
        phi_s = Input( shape=(STATE_SIZE,) )
        l_dense1 = Dense(16, activation='elu')(phi_s)
        out_value   = Dense(1, activation='linear')(l_dense1)
        model = Model(inputs=phi_s, outputs=out_value)
        return model

def _build_pi_model():
        # Builds the action graph
        # Inputs are 1. state 2. prob. distribution of an image
        # outputs are 1. \phi(a|s) and value V(s)
        phi_s = Input(shape=(STATE_SIZE,) )
        p_dist_i = Input( shape= (NUM_CLASSES,) )
        l_dense1 = Dense(32, activation='elu')(phi_s)
        l_dense2 = Dense(16, activation='elu')(p_dist_i)
        l_concat = concatenate([l_dense1, l_dense2], axis=1)
        out_action = Dense(1, activation='linear')(l_concat)
        model = Model(inputs=[p_dist_i, phi_s], outputs=out_action)
        return model




# from keras.datasets import mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
import numpy as np
class test_model():
    def __init__(self):
        self.phi_s_model = _build_phi_s_model()
        self.v_model = _build_v_model()
        self.pi_model = _build_pi_model()
        self.termination_action_model = _build_termination_action_model()

    def predict(self, s):
        probs, is_annotated = s
        # tf.reset_default_graph()
        model = _build_dynamic_model(self, is_annotated)
        # self.session.run(tf.global_variables_initializer())
                        
        #with self.default_graph.as_default():
        print(probs[0].reshape(1,10).shape)
        p, v = model.predict([probs[i].reshape(1,10) for i in xrange(probs.shape[0])])
        
        return p, v

    # x = np.random.rand(NUM_DATA, NUM_CLASSES)
    # is_annotated = set([0,3,5])
    # def func(arr):
        # s = sum(arr)
        # return [i*0.1/s for i in arr]
    # x = np.apply_along_axis(func, 1,x) 
    # model = _build_dynamic_model(self, is_annotated)
    # return x, model

if __name__=='__main__':
    # x, model = test_model()
    m = test_model()
    x = np.random.rand(NUM_DATA, NUM_CLASSES)
    is_annotated = set([0,3])
    p, v = m.predict((x, is_annotated)) 
    print(NUM_DATA)
    print(len(is_annotated))
    print('p',p)
    print('v',v)
    p, v = m.predict((x, is_annotated)) 
    print('p',p)
    print('v',v)
    is_annotated = set([0,3, 1])
    p, v = m.predict((x, is_annotated)) 
    print(NUM_DATA)
    print(len(is_annotated))
    print('p',p)
    print('v',v)
    # from keras.utils.layer_utils import print_summary
    # print_summary(model)
    # model.compile(optimizer='rmsprop', loss='binary_crossentropy', loss_weights=[1., 0.2])
