import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
class CNN:
    def __init__(self, input_shape, nb_classes):
        super(CNN, self).__init__()
        self.input_shape = input_shape
        self.nb_classes = nb_classes

    def build(self):
        i = tf.keras.layers.Input(shape=self.input_shape)  # input is amount of of samples
        x = tf.keras.layers.Conv2D(32, (3, 3), strides=2, activation="relu")(
            i)  # we are increasing the number of feature maps after each convolution
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), strides=2, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), strides=2, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(0.2)(x)  # along with dense layers we use Dropout for optimization
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        if self.nb_classes == 2:
            output = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)
        else:
            output = tf.keras.layers.Dense(self.nb_classes, activation="softmax")(x)
        model = tf.keras.Model(i, output)
        return model

    def compile(self, model):
        if self.nb_classes == 2:
            model.compile(loss="binary_crossentropy",
                          optimizer=tf.keras.optimizers.Adam(lr=0.001),
                          metrics=["accuracy"])
        else:
            model.compile(loss="categorical_crossentropy",  # one hot encoded
                          optimizer="adam",
                          metrics=["accuracy"])

    def train(self, model, with_generator, x_train, y_train, x_test, y_test, EPOCHS, BATCH_SIZE):
        es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1,
                                              patience=3, restore_best_weights=True)
        if with_generator == False:
            print("Training without data augmentation")
            history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, callbacks=[es])
            return history
        else:
            print("Training with data augmentation")
            data_generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1)
            train_generator = data_generator.flow(x_train, y_train, batch_size=BATCH_SIZE)
            steps_per_epoch = x_train.shape[0] // BATCH_SIZE
            history = model.fit_generator(train_generator, validation_data=(x_test, y_test),
                                          steps_per_epoch=steps_per_epoch, epochs=EPOCHS, callbacks=[es])
            return history