from base_classifier import BaseClassifier 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import yaml

class LeNet(BaseClassifier):
    '''
    Inspired from keras implementation of LeNet:
    https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
    
    Not a big change from this implementation.
    Only important thing to note here: there is a load_default_data method,
    and a load nondefault data method. These two methods together use preprocess_data
    to convert input to desired format, and sets training parameters.
    '''
    def __init__(self, config_path=None):
        if config_path:
            self.configs = yaml.load(open(config_path))
            # self.data = self.configs['data']
        else:
            self.batch_size = 128
            self.num_classes = 10
            self.epochs = 12
        self._get_default_data()
        self._create_model()
        self.ep = 0

    def _preprocess_data(self):
        img_rows, img_cols = 28, 28
        if K.image_data_format() == 'channels_first':
            self.x_train = self.x_train.reshape(self.x_train.shape[0], 1, img_rows, img_cols)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], 1, img_rows, img_cols)
            self.input_shape = (1, img_rows, img_cols)
        else:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], img_rows, img_cols, 1)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], img_rows, img_cols, 1)
            self.input_shape = (img_rows, img_cols, 1)
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)
    
    def _get_default_data(self):
        from keras.datasets import mnist


        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self._preprocess_data()

        print('x_train shape:', self.x_train.shape)
        print(self.x_train.shape[0], 'train samples')
        print(self.x_test.shape[0], 'test_samples')

    def _get_train_n(self):
        """
        This function gives number of classes and number of training samples
        """
        return self.x_train.shape[0], self.num_classes



    def _train(self, data=None):
        print 'Creating MNIST model...'
        x_train = self.x_train[list(self.is_annotated)]
        y_train = self.y_train[list(self.is_annotated)]
        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs,
                        validation_data=(self.x_test, self.y_test))

        return self._predict(x_train)

    def _predict(self, data=None):
        print 'Doing Predictions...'
        if data is None:
            return self.model.predict(self.x_test)
        else:
            return self.model.predict(data)

    def _evaluate(self, data=None):
        print 'Getting Accuracy...'
        if data is None:
            self._get_default_data()
        else:
            self._get_nondefault_data(data)
        score = self.model.evaluate(self.x_test, self.y_test)
        return score[1]

    def _create_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=keras.optimizers.Adadelta(), 
                        metrics=['accuracy'])
    
    def _reset(self):
        if self.model:
            del self.model

        self._create_model()

    def _save_model(self):
        self.model.save(self.configs['snapshot']+'/lenet-'+self.ep)
        self.ep+=1

def test_lenet():
    lenet = LeNet()
    lenet._train()
    pred_labels = lenet._predict()
    print pred_labels.shape
    accuracy = lenet._get_accuracy()
    print accuracy

if __name__ == '__main__':
    test_lenet()
