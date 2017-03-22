

class Classifier(object):

    @classmethod
    def get_classifier(self, classifier_name, config_path):
        subclasses = {cls.__class__.__name__:cls for cls in self.__subclasses__()}
        if classifier_name not in subclasses:
            raise Exception('Classifier not existed in:'+ str(names.keys())
        return subclasses[classifier_name](config_path)


    def train(self):
        self._train() 
    def predict(self):
        self._predict()
    def get_accuracy(self):
        self._get_accuracy()
        
    def _train(self):
        raise NotImplementedError
    def _predict(self):
        raise NotImplementedError
    def _get_accuracy(self):
        raise NotImplementedError





class LeNet(Classifier):

    def __init__(self, config_path):
        pass
    def _train(self):
        pass
    def _predict(self):
        pass
    def _get_accuracy(self):
        pass
    
   def _create_model(self):
       pass
