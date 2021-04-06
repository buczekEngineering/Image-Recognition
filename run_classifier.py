import tensorflow as tf
from CNNModel import CNN
from Monitoring import Monitoring
from Evaluation import Evaluation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Data import DataPrep
BATCH_SIZE = 32
EPOCHS = 10
with_generator = True


# shape of the input: (6000, 28,28) this is not the shape for CNN, it expects N x H x W x C, we must add color channel

data = DataPrep()
x_train, y_train, x_test, y_test = data.get_data()
x_train, x_test = data.preprocess_images(x_train, x_test)

input_shape = x_train[0].shape
y_train, y_test, nb_classes = data.one_hot_label(y_train, y_test)

#create a model
CNN = CNN(input_shape=input_shape, nb_classes=nb_classes)
base_model = CNN.build()
base_model.summary()
CNN.compile(base_model)
history = CNN.train(base_model, with_generator=True, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE)
base_model.save("base_model_CNN.h5")

monitor = Monitoring()
monitor.plot_accuracy(history)
monitor.plot_loss(history)

_, accuracy = base_model.evaluate(x_test,y_test)
print("Accuracy on test data: {:4.2f}%".format(accuracy*100))

print("==============TEST RESULTS============")

# eval = Evaluation(input_shape=input_shape, nb_classes=nb_classes)
# scores, histories = eval.cross_validation(n_splits=5, dataX=x_train, dataY=y_train, with_generator=False)
# print(scores, histories)

labels = ''' T-shirt/top
#  Trousers
#  Pullover
#  Dress
#  Coat
#  Sandal
#  Shirt
#  Sneaker
#  Bag
#  Ankle boot'''.split()
