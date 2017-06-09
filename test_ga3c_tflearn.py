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
# os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
# Added to make GPU not use maximum memory
# Note: CUDA_VISIBLE_DEVICES imposes an upper bound on total
# number of devices that could be used: https://github.com/tensorflow/tensorflow/issues/4566

from envs.active_learning_env import ActiveLearningEnv
import gym, time, random, threading
from gym.envs.registration import  register
# from envs.active_learning_env import ActiveLearningEnv
# register(id=ENV, entry_point='envs:ActiveLearningEnv')
import tflearn
from tflearn.layers import *
import subprocess
current_user = subprocess.check_output(['whoami']).strip()

config = yaml.load(open('config.yml'))
rl_args = config['RL']
classifier_args = config['classifiers']

locals().update(rl_args)
print sys.argv[2]
exit()
#---------
class Brain:
	train_queue = [ [], [], [], [], [] ]	# s, a, r, s', s' terminal mask
	lock_queue = threading.Lock()
	lock_graph = threading.Lock()
        with tf.device('/cpu:7'):
            config = tf.ConfigProto(device_count={'CPU':8}, 
                                        inter_op_parallelism_threads=8,
                                        intra_op_parallelism_threads=1)
            # config = tf.ConfigProto()
            # config.allow_soft_placement = True
            # config.log_device_placement = False
            # config.gpu_options.allow_growth=True

	def __init__(self):
		#K.manual_variable_initialization(True)

		# self.model = self._build_model()
                # tf.reset_default_graph()
                self.graph = tf.Graph()
                self.sess = tf.Session(graph=self.graph, config=Brain.config)
                self.rms_is_initialized = False
                self.iteration = {}
                self.train_writer = {}
                # self.test_writer = tf.summary.FileWriter(summaries_dir + '/test', self.sess.graph)
                # self.train_writer = tf.summary.FileWriter(summaries_dir + '/train', self.sess.graph)
                with self.sess.as_default():
                    with self.graph.as_default() as g:
                        l_input = input_data(shape=(None, NUM_CLASSES))
                        phi_s_tmp = self._build_phi_s_model(l_input, None)
                        # with tf.variable_scope('pi_model') as scope:
                        pi_tmp = self._build_pi_model(phi_s_tmp, l_input, None)
                        v_tmp = self._build_v_model(phi_s_tmp, None)
                        v_tmp = self._build_termination_action_model(phi_s_tmp, None)
                        with tf.variable_scope('graph_optimizer', reuse=None) as scope:
                            self.graph_optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99, name='GraphRMSProp')
                        try:
                            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                            # self.sess.run(init)
                            self.sess.run(tf.variables_initializer(tf.get_collection_ref('is_training')))
                        except:
                            init = tf.initialize_all_variables()
                        # self.sess.run(tf.global_variables_initializer())
                        self.sess.run(init)
                        print '+++++++  Done with Initializing'
                        # self.write2 = tf.summary.FileWriter(summaries_dir + '/train2', self.sess.graph)
                # self.train_writer = tf.summary.FileWriter(summaries_dir + '/train', self.sess.graph)
		# self.default_graph = tf.get_default_graph()


                ########### WHAT TO DO WITH THESE?
		# self.graph = self._build_graph(self.model)
		# self.default_graph.finalize()	# avoid modificationspu
		
        

        def _build_dynamic_model(self, is_annotated, inputs):
                with self.sess.as_default():
                    with self.graph.as_default() as g:
                        # create the network
                        not_annotated = set(range(NUM_DATA)).difference(is_annotated)
                        not_annotated = sorted(not_annotated) 
                        # delta from test_ga3c
                        # 1. Convert "Input" format of keras to input_data format of tflearn
                        # inputs = []
                        # for i in xrange(NUM_DATA):
                            # inputs.append(input_data(shape=(None, NUM_CLASSES)))

                        state_outputs = []
                            #share weights between different inputs
                        # with tf.contrib.framework.arg_scope([tflearn.variables.variable]) as scope:
                        iter_id = 1
                        print 'is_annotated', is_annotated
                        for a_i in is_annotated:
                            # print '----------------- build_annot:', a_i
                            phi_s_out = self._build_phi_s_model(inputs[a_i])
                            # scope.reuse_variables()
                            # phi_s_out = self.phi_s_model(inputs[a_i])
                            state_outputs.append(phi_s_out)
                            # tf.get_variable_scope().reuse_variables()
                            iter_id = iter_id + 1
                        #s_concat = merge(state_outputs, 'mean',axis=1)
                        phi_state = merge(state_outputs, 'mean',axis=0, name='avergae_pool_state')
                        phi_state = reshape(phi_state, (1, STATE_SIZE), name='reshape_state')
                        # print '+++++++++++++phi_state', phi_state
                        #TODO We could probably have phi_state for not_annotated instances

                        pi_outputs = []
                        # with tf.variable_scope('pi_model', reuse=True) as scope:
                        for a_i in not_annotated:
                            pi_i = self._build_pi_model(phi_state,inputs[a_i])#PI(a_i|phi(s))
                            pi_outputs.append(pi_i)
                            # scope.reuse_variables()

                        termination_action = self._build_termination_action_model(phi_state)
                        pi_outputs.append(termination_action)

                        action_concat = merge(pi_outputs, 'concat', axis=1, name='concatenate_PIs')
                        out_actions = tflearn.activations.softmax(action_concat)
                        out_value = self._build_v_model(phi_state)
                        return out_actions, out_value 

        def _build_graph(self, state, device):
            with self.lock_graph:
                with self.sess.as_default():
                    with self.graph.as_default() as g:
                        print '+++++++BUILD GRAPH++++++++++++'
                        with tf.device(device): 
                            summary_list = [] 
                            summary_dict= {} 
                            probs, is_annotated = state
                            # tf.reset_default_graph()
                            # model = self._build_dynamic_model(is_annotated, device)
                            inputs = [tf.placeholder(tf.float32, shape=(None, NUM_CLASSES), name='input_prob_tensor_{}_'.format(_)) for _ in xrange(NUM_DATA)]
                            # p, v = model(inputs)
                            p, v = self._build_dynamic_model(is_annotated, inputs)
                            # summary_list.append(tf.summary.scalar('LENGTH OF IS_ANNOTATED:', len(list(is_annotated))))
                            # summary_list.append(tf.Summary(value=[tf.Summary.Value(tag='Length of is_annotated', simple_value=len(list(is_annotated)))]))
                            print 'V vector::::::::::::::::::::::::', v
                            # summary_list.append(tf.summary.scalar('VALUE FUNCTION:', v[0][0]))
                            # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
                            # summary_writer.add_summary(summary, global_step=i)
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

                            summary_list.append(('Length of is_annotated',len(is_annotated)))
                            summary_dict['Value Function'] = v 
                            summary_dict['Log Prob'] = log_prob
                            summary_dict['Advantage'] = advantage
                            summary_dict['Loss Policy'] = loss_policy
                            summary_dict['Loss Value'] = loss_value
                            summary_dict['Total Loss'] = loss_total

                            # summary_list.append(tf.summary.scalar('LOG PROB: ', tf.reduce_max(log_prob)))
                            # summary_list.append(tf.summary.scalar('ADVANTAGE: ', tf.reduce_max(advantage)))
                            # summary_list.append(tf.summary.scalar('LOSS POLICY: ', tf.reduce_max(loss_policy)))
                            # summary_list.append(tf.summary.scalar('LOSS VALUE: ', tf.reduce_max(loss_value)))
                            # summary_list.append(tf.summary.scalar('TOTAL LOSS: ', tf.reduce_max(loss_total)))
                            # print 'TRAINABLE VARIABLES: ', ','.join([v.name for v in tf.trainable_variables()])
                            #with tf.variable_scope('optimizer', reuse=None) as scope:
                                #optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99, name='MyRMSProp_graph')
                                #tf.initialize_variables(optimizer)
                            minimize = self.graph_optimizer.minimize(loss_total)
                            # print 'GLOBAL VARIABLES: ', [v.name for v in tf.global_variables() ]
                            # rms_v = [v.name for v in tf.global_variables() if 'RMS' in v.name]
                            # print 'GLOBAL VARIABLES (RMS): ' 
                            # print 'GLOBAL VARIABLES (RMS): ', [v.name for v in tf.global_variables() if 'RMS' in v.name]
                            if not self.rms_is_initialized:
                                print '+++ INITIALIZING+++'
                                rms_vars= [vi for vi in tf.global_variables() if 'RMS' in vi.name]
                                # print '++++++++ initializing these:', rms_vars
                                init = tf.initialize_variables(rms_vars)
                                self.sess.run(init)
                                self.rms_is_initialized = True
                            # print 'LOCAL VARIABLES: ', [v.name for v in tf.local_variables() ]
                            #self.train_writer = tf.summary.FileWriter(summaries_dir + '/train', self.sess.graph)
                            # writer = tf.summary.FileWriter(summaries_dir + '/train', self.sess.graph)
                            return inputs, a_t, r_t, minimize, summary_list, summary_dict

                    
        def _build_predict_graph(self, state):
            pass
                # p_a, v = model(inputs)
                  

        def _build_termination_action_model(self, l_input, reuse=True):
            with self.sess.as_default():
                with self.graph.as_default() as g:
                    with tf.variable_scope('termination', reuse=reuse) as scope:
                        # Builds the state graph
                        # Input is an image class probabilities
                        # output is the state vector with size STATE_SIZE
                        # l_input = input_data( shape=(None, STATE_SIZE), name='input_termination_action_')
                        l_dense1 = fully_connected(l_input, 64, activation='elu', name='dense1_termination_action')
                        out_state = fully_connected(l_dense1, 1, activation='elu', name='out_state_')
                        #model = Model(inputs=l_input,outputs=out_state)
                        # model = tflearn.DNN(out_state, session=self.sess)
                        # model._make_predict_function()	# have to initialize before threading
                        # return model
                        return out_state

        def _build_phi_s_model(self, l_input, reuse=True):
            with self.sess.as_default():
                with self.graph.as_default() as g:
                    with tf.variable_scope('phi_s', reuse=reuse) as scope:
                        # Builds the state graph
                        # Input is an image class probabilities
                        # output is the state vector with size STATE_SIZE
                        # l_input = input_data(shape=(None, NUM_CLASSES) )
                        l_dense = fully_connected(l_input, 128, activation='elu',name='dens1_phi_s')
                        l_dense1 = fully_connected(l_dense, 64, activation='elu',name='dens2_phi_s')
                        out_state = fully_connected(l_dense1, STATE_SIZE, name='out_state_phi_s', activation='elu')
                        #model = Model(inputs=l_input,outputs=out_state)
                        # model = tflearn.DNN(out_state, session=self.sess)
                        # model._make_predict_function()	# have to initialize before threading
                        #return model
                        return out_state 

        def _build_v_model(self, phi_s, reuse=True):
            with self.sess.as_default():
                with self.graph.as_default() as g:
                    with tf.variable_scope('v', reuse=reuse) as scope:
                        l_dense1 = fully_connected(phi_s, 16, activation='elu', name='fc1_v')
                        out_value = fully_connected(l_dense1, 1, activation='linear', name='out_value')
                        return out_value

        def _build_pi_model(self,phi_s, p_dist_i, reuse=True):
            with self.sess.as_default():
                with self.graph.as_default() as g:
                    with tf.variable_scope('pi_model', reuse=reuse) as scope:
                        # Builds the action graph
                        # Inputs are 1. state 2. prob. distribution of an image
                        # outputs are 1. \phi(a|s) and value V(s)
                        # phi_s = input_data(shape=(None, STATE_SIZE) )
                        # p_dist_i = input_data( shape= (None, NUM_CLASSES) )
                        l_dense1 = fully_connected(phi_s, 32, activation='elu', name='dense1_pi_')
                        l_dense2 = fully_connected(p_dist_i, 16, activation='elu', name='dense2_pi_')
                        l_concat = merge([l_dense1, l_dense2], 'concat', axis=1, name='concat_pi_model')
                        out_action = fully_connected(l_concat, 1, activation='linear', name='out_action_')
                        return out_action

	def optimize(self, device, optimizer_id):
		if len(self.train_queue[0]) < MIN_BATCH:
			time.sleep(0)	# yield
			return
                
		with self.lock_queue:
			s_batch, a_batch, r_batch, s__batch, s_mask_batch = self.train_queue
			self.train_queue = [ [], [], [], [], [] ]

                if len(s_batch) > 5*MIN_BATCH: 
                        print("Optimizer alert! Minimizing batch of %d" % len(s_batch))
                
                # because model is different for each instance we can't process all in one batch; 
                # so we have to do it one by one, unless we change the implementation ...
                
                for i in xrange(len(a_batch)):
                        # print('ON BATCH:', i)
                        s = s_batch[i]
                        a = a_batch[i]
                        assert type(s[1])==set
                        is_annotated = s[1]
                        num_actions = 1+NUM_DATA-len(is_annotated)
                        a_cats = np.zeros(num_actions)	# turn action into one-hot representation
                        a_cats[a] = 1 
                        a = a_cats
                        r = r_batch[i]
                        s_= s__batch[i]
                        s_mask = s_mask_batch[i]


                        # tf.reset_default_graph()
                        # v = self.predict_v(s_, device)
                        v = np.array([10])
                        r = r + GAMMA_N * v * s_mask	# set v to 0 where s_ is terminal state
                        
                        # tf.reset_default_graph()
                        with self.sess.as_default():
                            with self.graph.as_default() as g:
                                print '#### FROM OPTIMIZER ####'
                                with tf.device(device):
                                    s_t, a_t, r_t, minimize, summary_list, summary_dict = self._build_graph(s, device)
                                print '#### DONE OPTIMIZER ####'
                                probs,is_annotated = s
                                # print 'OH', s_t[0]
                                # print 'PROBS', probs[0].reshape(1, -1) 
                                feed_dict = {s_t[i]:probs[i].reshape(1,-1) for i in xrange(probs.shape[0])}
                                feed_dict[a_t] = a.reshape(1,-1)
                                feed_dict[r_t] = r.reshape(1,-1)
                                # merged_train_summary = tf.summary.merge(summary_list)
                                # self.train_writer.add_summary(merged_train_summary,0)
                                if optimizer_id not in self.train_writer:
                                    print 'Creating Train Writer'
                                    self.train_writer[optimizer_id] =   tf.summary.FileWriter(OPTIMIZERS_SUMMARY_DIRS[optimizer_id], self.sess.graph)

                                    self.iteration[optimizer_id] = 0

                                # summary, loss = self.sess.run([merged_train_summary, minimize], feed_dict=feed_dict)
                                train_writer = self.train_writer[optimizer_id]
                                iteration = self.iteration[optimizer_id]
                                iteritems = [(key,val) for key, val in summary_dict.iteritems()]
                                # loss = self.sess.run([ minimize], feed_dict=feed_dict)
                                print iteritems
                                what_to_minimize = []
                                for key, val in iteritems:
                                    what_to_minimize.append(val)
                                what_to_minimize.append(minimize)
                                summaries= self.sess.run(what_to_minimize, feed_dict=feed_dict)
                                summaries = summaries[:-1]
                                # exit() 
                                # print '******************** THIS IS SUMMARY *********************', summary
                                for key,val in summary_list:
                                    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=val)])
                                    train_writer.add_summary(summary,iteration)
                                    print '------++++++++ SUMMARY: {0}:{1}'.format(key, val)
                                for i in xrange(len(summaries)):
                                    summary = tf.Summary(value=[tf.Summary.Value(tag=iteritems[i][0], simple_value=summaries[i])])
                                    print 'SUMMARY: {0}:{1}'.format(iteritems[i][0], summaries[i])
                                    train_writer.add_summary(summary,iteration)
                                
                                self.iteration[optimizer_id] += 1
                                # exit()

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
		# print(state)
                probs, is_annotated = state
                # tf.reset_default_graph()
                print '[device=',device,'] is_annotated =', is_annotated
                # model = self._build_dynamic_model(is_annotated, device)
                # self.session.run(tf.global_variables_initializer())
                # print 'probs', probs
                print NUM_DATA, probs.shape, '+++++++++++++'
                with self.sess.as_default():
                    with self.graph.as_default() as g:
                        p,v = None,None
                        with self.lock_graph:
                            with tf.device(device):
                                inputs = [tf.placeholder(tf.float32, shape=(None, NUM_CLASSES), name='input_prob_tensor_{}_'.format(_)) for _ in xrange(NUM_DATA)]
                                    # p, v = model(inputs)
                                feed_dict = {inputs[i]:probs[i].reshape(1,-1) for i in xrange(probs.shape[0])}
                                print '#### FROM PREDICT ####'
                                p, v = self._build_dynamic_model(is_annotated, inputs)
                                print '#### DONE WITH PREDICT ####'
                                # p, v = model.predict([probs[i].reshape(1,NUM_CLASSES) for i in xrange(probs.shape[0])])
                        p, v = self.sess.run([p,v], feed_dict=feed_dict)
			return p, v

	def predict_p(self, s, device):
                p, v = self.predict(s, device)
                return p

	def predict_v(self, s, device):
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
            return self.eps_end
		# if(frames >= self.eps_steps):
			# return self.eps_end
		# else:
			# return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps	# linearly interpolate

	def act(self, s):
		eps = self.getEpsilon()			
		# global frames; frames = frames + 1
                num_actions = NUM_DATA - len(s[1])+1
                # print 'in act, s', s
		if random.random() < eps:
			# return (random.randint(0, num_actions-1), True)
                        return (num_actions-1, True)

		else:
			# s = np.array([s])
			p = brain.predict_p(s, self.device)[0]

			# a = np.argmax(p)
			a = np.random.choice(num_actions, p=p)
			return (a, False)
	
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

	def __init__(self, render=False, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS, device=None, agent_id=None):
		threading.Thread.__init__(self)
		self.device = device
		self.render = render
                # self.env = ActiveLearningEnv(device='/gpu:0')
                self.agent_id = agent_id
                self.train_writer =   tf.summary.FileWriter(AGENT_SUMMARY_DIRS[self.agent_id], tf.get_default_graph())
                self.env = ActiveLearningEnv(device=device, summary_writer=self.train_writer)
		# self.env = gym.make(ENV)
                
                self.env.device = device
		self.agent = Agent(eps_start, eps_end, eps_steps,device)
                self.iteration = 0
	def runEpisode(self):
		s = self.env.reset()

		R = 0
		while True:         
			time.sleep(THREAD_DELAY) # yield 

			if self.render: self.env.render()

			a,is_random = self.agent.act(s)
                        #change action to environment type action
                        is_annotated = s[1]
                        num_actions = 1+NUM_DATA-len(is_annotated)
                        not_annotated = set(range(NUM_DATA)).difference(is_annotated)
                        not_annotated = sorted(not_annotated)
                        if a == num_actions-1:
                            action = (0, True)
                        else:
                            action = (not_annotated[a], False)
                        ######
                        summary = tf.Summary(value=[tf.Summary.Value(tag='action', simple_value=action[0])])
                        self.train_writer.add_summary(summary,self.iteration)
                        summary = tf.Summary(value=[tf.Summary.Value(tag='is_action_train', simple_value=5.0 if action[1] else -5.0)])
                        self.train_writer.add_summary(summary,self.iteration)
                        summary = tf.Summary(value=[tf.Summary.Value(tag='is_random_action', simple_value=5.0 if is_random else -5.0)])
                        self.train_writer.add_summary(summary,self.iteration)
                        self.iteration+=1
			s_, r, done, info = self.env.step(action)
                        summary = tf.Summary(value=[tf.Summary.Value(tag='reward', simple_value=r)])
                        self.train_writer.add_summary(summary,self.iteration)
                        probs,is_annotated = s_
                        assert probs.shape[0] == NUM_DATA
                        #TODO REMOVE NEXT 3 lines

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

	def __init__(self,device=None, optimizer_id=None):
		threading.Thread.__init__(self)
                self.device = device
                self.optimizer_id = optimizer_id
	def run(self):
		while not self.stop_signal:
			brain.optimize(self.device, self.optimizer_id)
	def stop(self):
		self.stop_signal = True

#-- main
# env_test = Environment(render=True, eps_start=0., eps_end=0., device='/gpu:0')
# env_test = Environment(render=True, eps_start=0., eps_end=0., device='/gpu:0')


# NUM_ACTIONS = env_test.env.action_space.n
NONE_STATE = np.zeros(STATE_SIZE)

if len(sys.argv) != 2:
    raise Exception('Experiment name not provided!')
EXP_NAME = sys.argv[1]
summaries_dir = '/local-scratch/'+current_user+'/rl-active/'+EXP_NAME+'/summaries/'

AGENT_SUMMARY_DIRS = []
for i in xrange(THREADS):
    AGENT_SUMMARY_DIRS.append(summaries_dir+'agent_'+str(i)+'/')
    os.system('mkdir '+AGENT_SUMMARY_DIRS[i]+' -p')
OPTIMIZERS_SUMMARY_DIRS =[]
for i in xrange(OPTIMIZERS):
    OPTIMIZERS_SUMMARY_DIRS.append(summaries_dir+'optimizer_'+str(i)+'/')
    os.system('mkdir '+OPTIMIZERS_SUMMARY_DIRS[i]+' -p')

brain = Brain()	# brain is global in A3C

# exit()
def gen_s():
    s = np.random.rand(NUM_DATA, NUM_CLASSES) 
    for i in xrange(s.shape[0]):
        s[i] = s[i]/sum(s[i])
    return s

# brain._build_dynamic_model(set([1,2]), '/gpu:0')
# brain._build_dynamic_model(gen_s(), set([2,1]))
# brain._build_graph((gen_s(), set([1,2])))
# exit()
# for i in xrange(8):
    # s,r,a,s_ = (gen_s(), set([0, 1])), -1, 2, (gen_s(), set([0,1,2]))
    # brain.train_push(s,r,a,s_)

# tf.reset_default_graph()
# a = brain._build_dynamic_model(s[1], '/gpu:0')
# a = brain._build_graph(s,'/gpu:0')
# a = brain.predict(s, '/gpu:0')
# brain.optimize('/gpu:0')
# envs = [Environment(device='/gpu:{}'.format(i),agent_id=i) for i in range(THREADS)]
envs = [Environment(device='/cpu:{}'.format(i),agent_id=i) for i in range(THREADS)]
opts = [Optimizer(device='/cpu:{}'.format(i+7), optimizer_id=i) for i in range(OPTIMIZERS)]
# opts = [Optimizer(device='/gpu:{}'.format(i), optimizer_id=i) for i in range(OPTIMIZERS)]

# op = Optimizer(device='/gpu:0')

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
# env_test.run()
