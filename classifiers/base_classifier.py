class BaseClassifier(object):

    # All subclasses must set the name and implement the following functions
    name = None

    def __init__(self, config_path=None, device=None):
        pass
        # self.config_path = config_path
        # self.is_annotated = set()

    def _train(self): 
        """
            This function trains the classifier on the training set 
        """
        raise NotImplementedError
    
    def _get_class_n(self):
        """
        This function gives number of classes and number of training samples
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

    def set_set_annotations(self, is_annotated):
        """
        This function sets a new annotation.
        """
        self.is_annotated = is_annotated

    def set_annotations(self, is_annotated):
        """
        This function adds a new annotation.
        """
        if hasattr(self, 'is_annotated'):
            self.is_annotated.update(is_annotated)
        else:
            self.is_annotated = is_annotated


    ###### no need to implement these in subclasses ###########
    def train(self):
        # Return: a numpy array of the predictions
        self.reset()
        probs = self._train()
        # self._save_model()
        return probs
    def reset(self):
        self._reset()
    def predict(self):
        return self._predict()
    def evaluate(self):
        return self._evaluate()
    def get_class_n(self):
        return self._get_class_n()
    def get_train_n(self):
        return self._get_train_n()
