
# from classifiers.classifier_factory import ClassifierFactory
# from classifier_factory import ClassifierFactory
from classifiers.classifier_factory import * 
# from random import random
import random
# dataset = set(range(50000))
# is_annotated = random.sample( dataset, 100)
is_annotated = random.sample( range(50000), 100)

lenet = ClassifierFactory.get_classifier('LeNet_TF', 'MNist', '', '/gpu:2')
lenet.set_annotations(is_annotated)
lenet.train()
acc = lenet.evaluate()

print 'ACCURACY:', acc

