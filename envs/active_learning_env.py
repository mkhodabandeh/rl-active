import logging
logger = logging.getLogger(__name__)

import numpy as np

import gym
from classifiers.classifier_factory import ClassifierFactory
from gym import error, spaces
from gym.utils import closer
# from dataset import Dataset
# env_closer = closer.Closer()

# Env-related abstractions

class ActiveLearningEnv(gym.Env):
    """The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    The main API methods that users of this class need to know are:
        step
        reset
        render
        close
        seed
    When implementing an environment, override the following methods
    in your subclass:
        _step
        _reset
        _render
        _close
        _seed
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
    The methods are accessed publicly as "step", "reset", etc.. The
    non-underscored versions are wrapper methods to which we may add
    functionality over time.
    """
    metadata = {'render.modes': []}
    reward_range = (-np.inf, np.inf)

    def __init__(self, classifier_name='LeNet_TF', dataset_name='MNist'):
        self.classifier_name = classifier_name
        self.dataset_name = dataset_name
        config_path = ''
        self.config_path = config_path
        self.classifier = ClassifierFactory.get_classifier(classifier_name,  dataset_name, config_path)
	self.n = self.classifier.get_train_n()
        self.k = self.classifier.get_class_n()
        self.action_space = spaces.Tuple((
                                  spaces.Discrete(self.n), # which instance to annotate (n -> size of the training set) 
                                  spaces.MultiBinary(1) # retrain the classifier using the new data 
                                   )) #TODO: verify this 
        print('number of instances:', self.n, 'number of classes:',self.k)
        self.probs = np.zeros((self.n,self.k))        
        self.best_val = 0
       	self.new_annotations = 0 
        self.previous_acc = 0
	# n_classes = self.dataset.n_classes
        self.max_annotations = self.n
        # self.observation_space = spaces.Tuple([spaces.Tuple([spaces.Box(0,1, 1) for i in n_classes]) for j in n]) # validation accuracy

    def _close(self):
	pass

    def _compute_reward(self, acc_gain=None):
	#TODO: define reward
	if acc_gain:
	    return acc_gain
	else:
            return -1

    def _step(self, action):
        label_i, do_train = action #action is a Tuple ( int, boolean)
	if do_train == True: 
	    self.classifier.set_annotations(self.is_annotated)
            self.probs = self.classifier.train()
            acc = self.classifier.evaluate()
            acc = acc[0]
            # print acc
	    # save best validation
	    acc_gain = acc - self.previous_acc 
            if acc > self.best_val:
                self.best_val = acc

            self.previous_acc = acc;
	    reward = self._compute_reward(acc_gain) 
            
	else:
            self.is_annotated.add(label_i)
            reward = self._compute_reward()
        done = len(self.is_annotated) == self.max_annotations
        return (self.probs.copy(), self.is_annotated.copy()), reward, done, None 
        

    def _reset(self): 
        print "+++++++++++++++++++++++INSIDE RESET"
        self.classifier = ClassifierFactory.get_classifier(self.classifier_name, self.dataset_name, self.config_path)
        print "++++++++++++++++++++CREATED THE CLASSIFIER"
<<<<<<< HEAD
	# self.is_annotated = set() 
        # self.best_val = 0
       # self.new_annotations = 0 
        self.is_annotated = set([0,1,]) 
        self.best_val = 0
        self.new_annotations = 0 
        self.probs = self.classifier.predict()
        print "++++++++++++++++++++ACQUIRED PROBABILITIES"
        return (self.probs.copy(), self.is_annotated.copy())

    def _render(self, mode='human', close=False): return
    def _seed(self, seed=None): return []

    def _render(self, mode='human', close=False): return
    def _seed(self, seed=None): return []

