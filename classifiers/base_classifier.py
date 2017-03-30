class BaseClassifier(object):

    # All subclasses must set the name and implement the following functions
    name = None

    def __init__(self, config_path=None):
        self.config_path = config_path

    def _train(self): 
        """
            This function trains the classifier on the training set 
        """
        raise NotImplementedError

    def _predict(self):
        """
        This function predicts the labels of test set
        """
        raise NotImplementedError
    
    def _get_accuracy(self):
        """
        This function returns the accuracy of predicted label. If predict() has not been called, this function should automatically call predict()
        """
        raise NotImplementedError

    ###### no need to implement these in subclasses ###########
    def train(self):
        self._train() 
    def predict(self):
        self._predict()
    def get_accuracy(self):
        self._get_accuracy()
        
