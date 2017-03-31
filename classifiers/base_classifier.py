class BaseClassifier(object):

    # All subclasses must set the name and implement the following functions
    name = None

    def __init__(self, config_path=None):
        self.config_path = config_path
        self.is_annotated = set()
    def _train(self): 
        """
            This function trains the classifier on the training set 
        """
        raise NotImplementedError
    
    def _get_train_n(self):
        """
        This function gives number of classes and number of training samples
        """
        raise NotImplementedError

    def _predict(self):
        """
        This function predicts the labels of test set
        """
        raise NotImplementedError
    
    def _evaluate(self):
        """
        This function returns the accuracy of predicted label. If predict() has not been called, this function should automatically call predict()
        """
        raise NotImplementedError

    def set_annotations(self, is_annotated):
        """
        This function adds a new annotation.
        """
        self.is_annotated += is_annotated
    


    ###### no need to implement these in subclasses ###########
    def train(self):
        self._reset()
        self._train()
        self._save_model()
    def predict(self):
        self._predict()
    def evaluate(self):
        self._evaluate()
    def get_train_n(self):
        self._get_train_n()
