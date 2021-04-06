import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Data import DataPrep
from Monitoring import *

data = DataPrep()
_, _, x_test, y_test = data.get_data()

monitor = Monitoring()
model_path = "base_model_CNN.h5"
base_model = tf.keras.models.load_model(model_path)

print("==============TESTING============")

predictions = base_model.predict(x_test)
prediction = np.argmax(predictions, axis =1)

monitor.print_metrics(y_test, y_pred=prediction)
