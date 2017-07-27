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

import os, yaml
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

from envs.active_learning_env import ActiveLearningEnv
import gym, time, random, threading
from gym.envs.registration import  register
# from envs.active_learning_env import ActiveLearningEnv
# register(id=ENV, entry_point='envs:ActiveLearningEnv')
import tflearn
from tflearn.layers import *
import subprocess
current_user = subprocess.check_output(['whoami']).strip()

CONFIG_PATH = 'config.yml'
config = yaml.load(open(CONFIG_PATH))
rl_args = config['RL']
shared_args = config['SHARED']
# classifier_args = config['classifiers']

locals().update(rl_args)
locals().update(shared_args)

#---------
class Brain:
	train_queue = [ [], [], [], [], [] ]	# s, a, r, s', s' terminal mask
	lock_queue = threading.Lock()

	def __init__(self):
                # config = tf.ConfigProto(device_count={'CPU':8}, 
                                        # inter_op_parallelism_threads=8,
                                        # intra_op_parallelism_threads=1)
		self.session = tf.Session()
		# K.set_session(self.session)
		# K.manual_variable_initialization(True)

		self.model = self._build_model()
		self.graph = self._build_graph(self.model)

		self.session.run(tf.global_variables_initializer())
		self.default_graph = tf.get_default_graph()

		self.default_graph.finalize()	# avoid modifications

        def _build_dynamic_model(self):
            ## is_annotated -> set
            ## not_annotated = set(range(NUM_DATA)).difference(is_annotated)
            
            ## inputs -> we are given NUM_DATA number of instances to annotated. They are across the batch dimension.
            ## input_size -> (NUM_DATA, NUM_CLASSES)
            inputs = tf.placeholder(tf.float32, shape=(None, NUM_CLASSES), name='input_prob_tensor')
            is_annotated_mask = tf.placeholder(tf.bool, shape=(NUM_DATA,), name='is_annotated_binary_vector')
            ## We need to split annotated ones from the rest using the boolean_mask function
            is_annotated_input = tf.boolean_mask(inputs, is_annotated_mask)
            not_annotated_input = tf.boolean_mask(inputs, tf.logical_not(is_annotated_mask))
            ## is_annotated_mask == [True, False, True, True, ...] K True and NUM_DATA-K False
            ## is_annotated_input size-> (K, NUM_CLASSES) 
            ## not_annotated_input size-> (NUM_DATA-K, NUM_CLASSES) 
            
            ## 
            phi_s_for_annotated = self._build_phi_s(is_annotated_input, for_annotated=True) 
            phi_s_for_not_annotated = self._build_phi_s(not_annotated_input, for_annotated=False) 
            
            ## s_l -> state of labeled set, s_u -> state of unlabeled set
            ## final state is the concatenation of [phi(s_l), phi(s_u), is_annotated]
            phi_state = merge([phi_s_for_annotated, phi_s_for_not_annotated, tf.cast(is_annotated_mask, tf.float32)], 
                                'concat',axis=1, name='final_phi_state')
            assert tf.shape(phi_state) == (1, 2*STATE_SIZE+NUM_DATA)
            # phi_state = reshape(phi_state, (1, 2*STATE_SIZE+NUM_DATA), name='reshape_state')

            termination_action = self._build_termination_action_model(phi_state)
            policy_scores = self._build_pi_model(phi_state,inputs, termination_action)#PI(a_i|phi(s))
            out_actions= tflearn.activations.softmax(policy_scores)

            out_value = self._build_v_model(phi_state)
            all_inputs = [inputs, is_annotated_mask] 
            all_outputs = [out_actions, out_value]
            return all_inputs, all_outputs 


        # def _build_phi_s_model(self, l_input reuse=True):
        def _build_phi_s_model(self, l_input, for_annotated):
                # Builds the state graph
                # Input is a tensor containing image class probabilities (NUM_DATA, NUM_CLASSES)
                # output is the state vector with size STATE_SIZE
                # l_input = input_data(shape=(None, NUM_CLASSES) )
            name = 'phi_s_for_annotated' if for_annotated else 'phi_s_for_not_annotated'
            with tf.variable_scope(name, reuse=False) as scope:
                l_dense = fully_connected(l_input, 128, activation='elu',name='dense1_phi_s')
                l_dense1 = fully_connected(l_dense, 64, activation='elu',name='dense2_phi_s')
                out_state = fully_connected(l_dense1, STATE_SIZE, name='out_state_phi_s', activation='elu')
                out_state = tf.reduce_mean(out_state, 0, name='average_pool_state') #Average pooling across the batch dimension
                phi_s = reshape(out_state, (1, STATE_SIZE), name='phi_s')
                return phi_s 

        def _build_termination_action_model(self, phi_s):
            with tf.variable_scope('termination', reuse=False) as scope:
                # Builds the termination action model 
                # Input is the phi(s) 
                # output is one number, which is the score for terminating the episode (train the classifier) 
                l_dense1 = fully_connected(phi_s, 64, activation='elu', name='dense1_termination_action')
                score = fully_connected(l_dense1, 1, activation='elu', name='out_state_')
                return score 

        def _build_pi_model(self,phi_s, inputs, termination):
            ## Builds the policy network 
            ## Inputs are 1-state 2-prob. distribution of an image 3-termination score
            ## outputs are 1. \phi(a|s) and value V(s)
            ## phi_s = input_data(shape=(1, STATE_SIZE) )
            ## p_dist_i = input_data( shape= (None, NUM_CLASSES) )
            with tf.variable_scope('pi_model', reuse=False) as scope:
                l_dense1 = fully_connected(phi_s, 32, activation='elu', name='dense1_pi_')
                l_dense1 = tf.tile(l_dense1, [tf.shape(inputs)[0], 1], 'dense1_pi_replicated') #replicate phi_s for NUM_DATA times along the batch dimension
                l_dense2 = fully_connected(inputs, 32, activation='elu', name='dense2_pi_')
                l_concat = merge([l_dense1, l_dense2], 'concat', axis=1, name='concat_pi_model')
                out_action = fully_connected(l_concat, 1, activation='linear', name='out_action_')
                out_action_reshaped = tf.reshape(out_action, [1, tf.shape(inputs)[0]], 'out_action_reshaped_') #
                policy_score = tf.concat([out_action_reshaped, termination), axis=1, name='out_action_and_terminate')
                return policy_score 

        def _build_v_model(self, phi_s, reuse=True):
            with tf.variable_scope('v', reuse=reuse) as scope:
                l_dense1 = fully_connected(phi_s, 16, activation='elu', name='fc1_v')
                out_value = fully_connected(l_dense1, 1, activation='linear', name='out_value')
                return out_value

	def _build_graph(self, all_inputs, all_outputs):
                inputs, is_annotated_mask = all_inputs
		s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATE))
                NUM_ACTIONS = NUM_DATA+1
                a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
		r_t = tf.placeholder(tf.float32, shape=(None, 1)) # not immediate, but discounted n step reward
		
                p, v = all_outputs
		# p, v = model(s_t)

		log_prob = tf.log( tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10, name='log_prob')
		advantage = r_t - v

		loss_policy = - log_prob * tf.stop_gradient(advantage)									# maximize policy
		loss_value  = LOSS_V * tf.square(advantage)												# minimize value error
		entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)	# maximize entropy (regularization)

		loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

		optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
		minimize = optimizer.minimize(loss_total)

		return s_t, a_t, r_t, minimize

	def optimize(self):
		if len(self.train_queue[0]) < MIN_BATCH:
			time.sleep(0)	# yield
			return

		with self.lock_queue:
			if len(self.train_queue[0]) < MIN_BATCH:	# more thread could have passed without lock
				return 									# we can't yield inside lock

			s, a, r, s_, s_mask = self.train_queue
			self.train_queue = [ [], [], [], [], [] ]

		s = np.vstack(s)
		a = np.vstack(a)
		r = np.vstack(r)
		s_ = np.vstack(s_)
		s_mask = np.vstack(s_mask)

		if len(s) > 5*MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))

		v = self.predict_v(s_)
		r = r + GAMMA_N * v * s_mask	# set v to 0 where s_ is terminal state
		
		s_t, a_t, r_t, minimize = self.graph
		self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})

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

	def predict(self, s):
		with self.default_graph.as_default():
			p, v = self.model.predict(s)
			return p, v

	def predict_p(self, s):
		with self.default_graph.as_default():
			p, v = self.model.predict(s)		
			return p

	def predict_v(self, s):
		with self.default_graph.as_default():
			p, v = self.model.predict(s)		
			return v

#---------
frames = 0
class Agent:
	def __init__(self, eps_start, eps_end, eps_steps):
		self.eps_start = eps_start
		self.eps_end   = eps_end
		self.eps_steps = eps_steps

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
			p = brain.predict_p(s)[0]

			# a = np.argmax(p)
			a = np.random.choice(NUM_ACTIONS, p=p)

			return a
	
	def train(self, s, a, r, s_):
		def get_sample(memory, n):
			s, a, _, _  = memory[0]
			_, _, _, s_ = memory[n-1]

			return s, a, self.R, s_

		a_cats = np.zeros(NUM_ACTIONS)	# turn action into one-hot representation
		a_cats[a] = 1 

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

	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):
		while not self.stop_signal:
			brain.optimize()

	def stop(self):
		self.stop_signal = True

#-- main
env_test = Environment(render=True, eps_start=0., eps_end=0.)
NUM_STATE = env_test.env.observation_space.shape[0]
NUM_ACTIONS = env_test.env.action_space.n
NONE_STATE = np.zeros(NUM_STATE)

brain = Brain()	# brain is global in A3C

envs = [Environment() for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]

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

