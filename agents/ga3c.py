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

from keras.models import *
from keras.layers import *
from keras import backend as K

#-- constants
ENV = 'CartPole-v0'

RUN_TIME = 30
THREADS = 8
OPTIMIZERS = 2
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.4
EPS_STOP  = .15
EPS_STEPS = 75000

MIN_BATCH = 32
LEARNING_RATE = 5e-3

LOSS_V = .5			# v loss coefficient
LOSS_ENTROPY = .01 	# entropy coefficient

STATE_SIZE = 128
NUM_CLASSES = 10
NUM_DATA = 1000
#---------
class Brain:
	train_queue = [ [], [], [], [], [] ]	# s, a, r, s', s' terminal mask
	lock_queue = threading.Lock()

	def __init__(self):
		self.session = tf.Session()
		K.set_session(self.session)
		K.manual_variable_initialization(True)

		# self.model = self._build_model()
                self.phi_s_model = self._build_phi_s_model()
                self.v_model = self._build_v_model()
                self.pi_model = self._build_pi_model()
                self.termination_train_model = self._build_termination_action_model()

		self.session.run(tf.global_variables_initializer())
		self.default_graph = tf.get_default_graph()


                ########### WHAT TO DO WITH THESE?
		# self.graph = self._build_graph(self.model)
		# self.default_graph.finalize()	# avoid modifications
        

        def _build_dynamic_model(self, is_annotated):
                # create the network
                not_annotated = set(range(NUM_DATA)).difference(is_annotated)
                inputs = []
                for i in xrange(NUM_DATA):
                    # inputs.append(tf.placeholder(tf.float32, shape=(None, NUM_CLASSES)))
                    inputs.append(Input(shape=(NUM_CLASSES,)))
                state_outputs = []
                for a_i in is_annotated:
                    state_outputs.append(self.phi_s_model(inputs[a_i]))
                s_concat = concatenate(state_outputs, axis=0)
                s_concat = Reshape((1,STATE_SIZE))(s_concat)
                phi_state = GlobalAveragePooling1D()(s_concat)
                
                #TODO We could probably have phi_state for not_annotated instances

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

        def _build_graph(self, state, device='/gpu:0'):
        
		with tf.device(device): 
			probs, is_annotated = state
			tf.reset_default_graph()
			model = self._build_dynamic_model(is_annotated)
			inputs = [tf.placeholder(tf.float32, shape=(None, NUM_CLASSES)) for _ in NUM_DATA]
			p, v = model(inputs)

			assert NUM_DATA-len(is_annotated) > 0, 'Nothing to annotate'
			a_t = tf.placeholder(tf.float32, shape=(None, NUM_DATA-len(is_annotated))
			r_t = tf.placeholder(tf.float32, shape=(None, 1)) # discounted n step reward
			s_t = [tf.placeholder(tf.float32, shape=(None, NUM_CLASSES) for i in xrange(NUM_DATA)]
			
			log_prob = tf.log( tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
			advantage = r_t - v

			loss_policy = - log_prob * tf.stop_gradient(advantage)	# maximize policy
			loss_value  = LOSS_V * tf.square(advantage)  # minimize value error
			entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)	# maximize entropy (regularization)

			loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

			optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
			minimize = optimizer.minimize(loss_total)

			return s_t, a_t, r_t, minimize

                    
        def _build_predict_graph(self, state):
                p_a, v = model(inputs)
                  

        def _build_termination_action_model(self):
                # Builds the state graph
                # Input is an image class probabilities
                # output is the state vector with size STATE_SIZE
                l_input = Input( shape=(STATE_SIZE,) )
                l_dense1 = Dense(64, activation='elu')(l_input)
                out_state = Dense(1, activation='elu')(l_dense1)
                model = Model(inputs=l_input,outputs=out_state)
                # model._make_predict_function()	# have to initialize before threading
                return model

        def _build_phi_s_model(self):
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

        def _build_v_model(self):
                phi_s = Input( shape=(STATE_SIZE,) )
                l_dense1 = Dense(16, activation='elu')(phi_s)
                out_value   = Dense(1, activation='linear')(l_dense1)
                model = Model(inputs=phi_s, outputs=out_value)
                return model

        def _build_pi_model(self):
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


	def optimize(self, device):
		if len(self.train_queue[0]) < MIN_BATCH:
			time.sleep(0)	# yield
			return
                
		with self.lock_queue:
			s_batch, a_batch, r_batch, s__batch, s_mask_batch = self.train_queue
			self.train_queue = [ [], [], [], [], [] ]

                #TODO  Do the following
                        # 1-Turn state and action to np.array
                        # 2-Build the graph
                        # 3-define loss -> minimize
                        # 4-Feed run session on minimize
                if len(s_batch) > 5*MIN_BATCH: 
                        print("Optimizer alert! Minimizing batch of %d" % len(s_batch))
                
                # because model is different for each instance we can't process all in one batch; 
                # so we have to do it one by one, unless we change the implementation ...
                
                for i in xrange(len(a_batch)):
                        s = s_batch[i]
                        a = a_batch[i]
                        r = r_batch[i]
                        s_= s__batch[i]
                        s_mask = s_mask_batch[i]


                        v = self.predict_v(s_)
                        r = r + GAMMA_N * v * s_mask	# set v to 0 where s_ is terminal state
                        
                        s_t, a_t, r_t, minimize = self._build_graph(s, device)
                        probs,is_annotated = s
                        feed_dict = {s_t[i]:probs[i].reshape((1,NUM_CLASSES)) for i in xrange(probs.shape[0])}
                        feed_dict[a_t] = a
                        feed_dict[r_t] = r
                        self.session.run(minimize, feed_dict=feed_dict)
                
	def train_push(self, s, a, r, s_):
		with self.lock_queue:
			self.train_queue[0].append(s)
			self.train_queue[1].append(a)
			self.train_queue[2].append(r)

			if s_ is None:
				self.train_queue[3].append(NONE_STATE)
				self.train_queue[4].append(0.)
			else:	
				self.train_queue[3].append(s_)
				self.train_queue[4].append(1.)

	def predict(self, state, device):
                probs, is_annotated = state
                tf.reset_default_graph()
                model = self._build_dynamic_model(is_annotated)
                self.session.run(tf.global_variables_initializer())
                                
		with self.default_graph.as_default():
			p, v = self.model.predict([probs[i] for i in xrange(probs.shape[0])])
			return p, v

	def predict_p(self, s, device):
                p, v = self.predict(s, device)
                return p

	def predict_v(self, s):
                p, v = self.predict(s, device)
                return v
#---------
frames = 0
class Agent:
	def __init__(self, eps_start, eps_end, eps_steps, device):
		self.eps_start = eps_start
		self.eps_end   = eps_end
		self.eps_steps = eps_steps
		self.device = device
		self.memory = []	# used for n_step return
		self.R = 0.

	def getEpsilon(self):
		if(frames >= self.eps_steps):
			return self.eps_end
		else:
			return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps	# linearly interpolate

	def act(self, s):
		eps = self.getEpsilon()			
		global frames; frames = frames + 1

		if random.random() < eps:
			return random.randint(0, NUM_ACTIONS-1)

		else:
			s = np.array([s])
			p = brain.predict_p(s, self.device)[0]

			# a = np.argmax(p)
			a = np.random.choice(NUM_ACTIONS, p=p)

			return a
	
	def train(self, s, a, r, s_):
		def get_sample(memory, n):
			s, a, _, _  = memory[0]
			_, _, _, s_ = memory[n-1]

			return s, a, self.R, s_

		# a_cats = np.zeros(NUM_ACTIONS)	# turn action into one-hot representation
		# a_cats[a] = 1 
                
                a_cats = a # doesn't have to turn it into one-hot representation                

		self.memory.append( (s, a_cats, r, s_) )

		self.R = ( self.R + r * GAMMA_N ) / GAMMA

		if s_ is None:
			while len(self.memory) > 0:
				n = len(self.memory)
				s, a, r, s_ = get_sample(self.memory, n)
				brain.train_push(s, a, r, s_)

				self.R = ( self.R - self.memory[0][2] ) / GAMMA
				self.memory.pop(0)		

			self.R = 0

		if len(self.memory) >= N_STEP_RETURN:
			s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
			brain.train_push(s, a, r, s_)

			self.R = self.R - self.memory[0][2]
			self.memory.pop(0)	
	
	# possible edge case - if an episode ends in <N steps, the computation is incorrect
		
#---------
class Environment(threading.Thread):
	stop_signal = False

	def __init__(self, render=False, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS):
		threading.Thread.__init__(self)

		self.render = render
		self.env = gym.make(ENV)
		self.agent = Agent(eps_start, eps_end, eps_steps)

	def runEpisode(self):
		s = self.env.reset()

		R = 0
		while True:         
			time.sleep(THREAD_DELAY) # yield 

			if self.render: self.env.render()

			a = self.agent.act(s)
			s_, r, done, info = self.env.step(a)

			if done: # terminal state
				s_ = None

			self.agent.train(s, a, r, s_)

			s = s_
			R += r

			if done or self.stop_signal:
				break

		print("Total R:", R)

	def run(self):
		while not self.stop_signal:
			self.runEpisode()

	def stop(self):
		self.stop_signal = True

#---------
class Optimizer(threading.Thread):
	stop_signal = False

	def __init__(self,device='/gpu:0'):
		threading.Thread.__init__(self)
                self.device = device

	def run(self):
		while not self.stop_signal:
			brain.optimize(self.device)

	def stop(self):
		self.stop_signal = True

#-- main
env_test = Environment(render=True, eps_start=0., eps_end=0.)
NUM_STATE = env_test.env.observation_space.shape[0]
NUM_ACTIONS = env_test.env.action_space.n
NONE_STATE = np.zeros(NUM_STATE)

brain = Brain()	# brain is global in A3C

envs = [Environment('/gpu:0') for i in range(THREADS)]
# envs = [Environment('/gpu:'+str(i%4)) for i in range(THREADS)]
opts = [Optimizer('/gpu:0') for i in range(OPTIMIZERS)]
# opts = [Optimizer('/gpu:'+str(i%4)) for i in range(OPTIMIZERS)]

for o in opts:
	o.start()

for e in envs:
	e.start()

time.sleep(RUN_TIME)

for e in envs:
	e.stop()
for e in envs:
	e.join()

for o in opts:
	o.stop()
for o in opts:
	o.join()

print("Training finished")
env_test.run()
