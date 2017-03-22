from lenet import LeNet

class Classifier(object):
    CLASSIFIERS = [LeNet]

    @classmethod
    def get_classifier(self, classifier_name, config_path):
        for cls in self.CLASSIFIERS:
            if cls.name == classifier_name:
                return cls(config_path)
        raise Exception('Classifier, {0}, is not existed in: {1}'.format(classifier_name ,str(self.get_classifiers_list())) 
       
    @classmethod
    def get_classifiers_list(self):
        return [cls.name for cls in self.CLASSIFIERS]

    def __init__(self):
        raise NotImplementedError
    

