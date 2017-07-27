# OpenGym CartPole-v0 with A3C on GPU
# -----------------------------------
#
# A3C implementation with GPU optimizer threads.
# 
# Made as part of blog series Let's make an A3C, available at
# https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/
#
# author: Jaromir Janisch, 2017

import sys
sys.stdout.flush()
import numpy as np
sys.stdout.flush()


import tensorflow as tf

import gym, time, random, threading
from gym.envs.registration import  register
# from envs.active_learning_env import ActiveLearningEnv
# register(id=ENV, entry_point='envs:ActiveLearningEnv')
from tflearn.layers import *
import tflearn

#-- constants
ENV = 'ActiveLearningEnv-v0'
gym.make(ENV)
# exit()
RUN_TIME = 5 
THREADS = 1
OPTIMIZERS = 1
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.4
EPS_STOP  = .15
EPS_STEPS = 75000

# MIN_BATCH = 32
MIN_BATCH = 5 
LEARNING_RATE = 5e-3

LOSS_V = .5			# v loss coefficient
LOSS_ENTROPY = .01 	# entropy coefficient

STATE_SIZE = 128
NUM_CLASSES = 10
NUM_DATA = 20 
#---------
class Brain:
	train_queue = [ [], [], [], [], [] ]	# s, a, r, s', s' terminal mask
	lock_queue = threading.Lock()

	def __init__(self):
		#K.manual_variable_initialization(True)

		# self.model = self._build_model()
                # tf.reset_default_graph()
                # self.phi_s_model = self._build_phi_s_model()
                self.graph = tf.Graph()
                self.sess = tf.Session(graph=self.graph)
                self.pi_model = self._build_pi_model()
                with self.sess.as_default():
                    with self.graph.as_default() as g:
                        self.sess.run(tf.global_variables_initializer())
		# self.default_graph = tf.get_default_graph()


                ########### WHAT TO DO WITH THESE?
		# self.graph = self._build_graph(self.model)
		# self.default_graph.finalize()	# avoid modifications
        

        def _build_dynamic_model(self, is_annotated, device):
            with self.sess.as_default():
                with self.graph.as_default() as g:
                    # create the network
                    not_annotated = set(range(NUM_DATA)).difference(is_annotated)
                    inputs = []
                    # delta from test_ga3c
                    # 1. Convert "Input" format of keras to input_data format of tflearn
                    for i in xrange(NUM_DATA):
                        # inputs.append(tf.placeholder(tf.float32, shape=(None, NUM_CLASSES)))
                        inputs.append(input_data(shape=(None, NUM_CLASSES)))

                    state_outputs = []
                        #share weights between different inputs
                    # with tf.contrib.framework.arg_scope([tflearn.variables.variable]) as scope:
                    iter_id = 1
                    with tf.variable_scope('phi_s') as scope:
                        for a_i in is_annotated:
                            phi_s_out = self._build_phi_s_model(inputs[a_i])
                            state_outputs.append(phi_s_out)
                            scope.reuse_variables()
                            # tf.get_variable_scope().reuse_variables()
                            print 'ITER: '+ str(iter_id)
                            iter_id = iter_id + 1
                    #s_concat = merge(state_outputs, 'mean',axis=1)
                    phi_state = tflearn.merge(state_outputs, 'mean',axis=1)
                    #phi_state = GlobalAveragePooling1D(name='global_max_pool_phi_s')(s_concat)
        
                    return tflearn.DNN(phi_state, session=self.sess)
                    #TODO We could probably have phi_state for not_annotated instances
        
                    pi_outputs = []
                    with tf.variable_scope('pi_model') as scope:
                        for a_i in not_annotated:
                            # print inputs[a_i]
                            # print phi_state
                            # print self.pi_model
                            pi_i = self.pi_model([inputs[a_i], phi_state])#PI(a_i|phi(s))
                            pi_outputs.append(pi_i)
                            scope.reuse_variables()

                    termination_action = self.termination_action_model(phi_state)
                    pi_outputs.append(termination_action)

                    action_concat = merge(pi_outputs, 'concat', axis=1, name='concatenate_PIs')
                    out_actions = tflearn.activations.sotfmax(action_concat, name='action_output')
                    out_value = self.v_model(phi_state)
                    #TODO:Check with mehran about merge type
                    final_model = tflearn.merge([out_actions, out_value], 'elemwise_sum')
                    #final_model._make_predict_function()

                    return final_model

        def _build_graph(self, state, device='/gpu:0'):
            with self.sess.as_default():
                with self.graph.as_default() as g:
            
                    # with tf.device(device): 
                    probs, is_annotated = state
                    # tf.reset_default_graph()
                    model = self._build_dynamic_model(is_annotated, device)
                    inputs = [tf.placeholder(tf.float32, shape=(None, NUM_CLASSES), name='input_prob_tensor_{}_'.format(_)) for _ in xrange(NUM_DATA)]
                    v = model(inputs)

                    return v
                    assert NUM_DATA-len(is_annotated) > 0, 'Nothing to annotate'
                    a_t = tf.placeholder(tf.float32, shape=(None, 1+NUM_DATA-len(is_annotated)), name='a_t')
                    r_t = tf.placeholder(tf.float32, shape=(None, 1), name='r_t') # discounted n step reward
                    # s_t = [tf.placeholder(tf.float32, shape=(None, NUM_CLASSES), name='s_t_{}_'.format(i)) for i in xrange(NUM_DATA)]

                    log_prob = tf.log( tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10, name='log_prob')
                    advantage = r_t - v

                    loss_policy = - log_prob * tf.stop_gradient(advantage)	# maximize policy
                    loss_value  = LOSS_V * tf.square(advantage)  # minimize value error
                    entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True, name='entropy')	# maximize entropy (regularization)

                    loss_total = tf.reduce_mean(loss_policy + loss_value + entropy, name='loss_total')

                    optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99, name='MyRMSProp')
                    #tf.initialize_variables(optimizer)
                    minimize = optimizer.minimize(loss_total)

                    return inputs, a_t, r_t, minimize

                    
        def _build_predict_graph(self, state):
                p_a, v = model(inputs)
                  

        def _build_termination_action_model(self):
            with self.sess.as_default():
                with self.graph.as_default() as g:
                    # Builds the state graph
                    # Input is an image class probabilities
                    # output is the state vector with size STATE_SIZE
                    l_input = input_data( shape=(None, STATE_SIZE), name='input_termination_action_')
                    l_dense1 = fully_connected(l_input, 64, activation='elu', name='dense1_termination_action')
                    out_state = fully_connected(l_dense1, 1, activation='elu', name='out_state_')
                    #model = Model(inputs=l_input,outputs=out_state)
                    model = tflearn.DNN(out_state, session=self.sess)
                    # model._make_predict_function()	# have to initialize before threading
                    return model
                    #return out_state

        def _build_phi_s_model(self, l_input):
            with self.sess.as_default():
                with self.graph.as_default() as g:
                    with tf.variable_scope('phi_s') as scope:
                        # Builds the state graph
                        # Input is an image class probabilities
                        # output is the state vector with size STATE_SIZE
                        #l_input = input_data(shape=(None, NUM_CLASSES) )
                        l_dense = fully_connected(l_input, 128, activation='elu',name='dens1_phi_s', scope='dens1_phi_s')
                        l_dense1 = fully_connected(l_dense, 64, activation='elu',name='dens2_phi_s', scope='dens2_phi_s')
                        out_state = fully_connected(l_dense1, STATE_SIZE, activation='elu')
                        #model = Model(inputs=l_input,outputs=out_state)
                        #model = tflearn.DNN(out_state, session=self.session)
                        # model._make_predict_function()	# have to initialize before threading
                        #return model
                        return out_state

        def _build_v_model(self):
            with self.sess.as_default():
                with self.graph.as_default() as g:
                    #phi_s = Input( shape=(STATE_SIZE,) )
                    phi_s = input_data( shape=(None, STATE_SIZE))
                    l_dense1 = fully_connected(phi_s, 16, activation='elu', name='fc1_v')
                    out_value = fully_connected(l_dense1, 1, activation='linear')
                    model = tflearn.DNN(out_value, session=self.sess)
                    return model
                    #return out_value

        def _build_pi_model(self):
            with self.sess.as_default():
                with self.graph.as_default() as g:
                    with tf.variable_scope('pi_model') as scope:
                        # Builds the action graph
                        # Inputs are 1. state 2. prob. distribution of an image
                        # outputs are 1. \phi(a|s) and value V(s)
                        phi_s = input_data(shape=(None, STATE_SIZE) )
                        p_dist_i = input_data( shape= (None, NUM_CLASSES) )
                        l_dense1 = fully_connected(phi_s, 32, activation='elu', name='dense1_pi_')
                        l_dense2 = fully_connected(p_dist_i, 16, activation='elu', name='dense2_pi_')
                        l_concat = merge([l_dense1, l_dense2], 'concat', axis=1, name='concat_pi_model')
                        out_action = fully_connected(l_concat, 1, activation='linear', name='out_action_')
                        model = tflearn.DNN(out_action, session=self.sess)
                        #return out_action
                        return model


	def predict(self, state, device):
		# print(state)
                probs, is_annotated = state
                # tf.reset_default_graph()
                model = self._build_dynamic_model(is_annotated, device)
                # self.session.run(tf.global_variables_initializer())
                                
                with self.sess.as_default():
                    with self.graph.as_default():
                        v = model.predict([probs[i].reshape(1,NUM_CLASSES) for i in xrange(probs.shape[0])])
                        return v 

def gen_s():
        s = np.random.rand(NUM_DATA, NUM_CLASSES) 
        for i in xrange(s.shape[0]):
            s[i] = s[i]/sum(s[i])
        return s

brain = Brain()
brain.predict((gen_s(), set([1,2])), '/gpu:0')
# brain._build_graph((gen_s(), set([1,2]))
# brain._build_dynamic_model( set([1,2,3]), '/gpu:0')
