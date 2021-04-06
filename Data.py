import tensorflow as tf

class DataPrep:
    def __init__(self):
        super(DataPrep).__init__()

    def preprocess_images(self,x_train, x_test):
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        x_train = tf.expand_dims(x_train, axis=-1)
        x_test = tf.expand_dims(x_test, axis=-1)
        return x_train, x_test

    def one_hot_label(self, y_train, y_test):
        nb_classes = len(set(y_train))
        y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
        y_test = tf.keras.utils.to_categorical(y_test, nb_classes)
        return y_train, y_test, nb_classes

    def get_data(self):
        data = tf.keras.datasets.fashion_mnist.load_data()
        (x_train, y_train), (x_test, y_test) = data

        return x_train, y_train, x_test, y_test