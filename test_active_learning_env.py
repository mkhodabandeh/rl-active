from envs.active_learning_env import ActiveLearningEnv
import threading
import gym, time
ENV = 'ActiveLearningEnv-v0'
RUN_TIME = 10
class Environment(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.env = ActiveLearningEnv(classifier_name='Cifar10')
        # self.env = gym.make(ENV) 
    def run(self):
        # pass
        self.env.classifier.is_annotated = set(range(1000))
        self.env.classifier.train()
        self.env.classifier.predict()
        print self.env.classifier.evaluate()
        # self.env.classifier.reset()
        # self.env.classifier.train()

if __name__=='__main__':
    a = Environment()
    # b = Environment()
    a.start()
    # time.sleep(RUN_TIME)
    # b.start()
