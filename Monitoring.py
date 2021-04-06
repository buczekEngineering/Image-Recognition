from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np

class Monitoring:
    def __init__(self):
        super(Monitoring, self).__init__()

    def print_confusion_matrix(self, y_test, y_pred, title="Confusion Matrix"):
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        return cm

    def print_metrics(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        print("Evaluation on test data")
        print("Accuracy: {}%".format(accuracy*100))
        print("Precision: {}%".format(precision*100))
        print("F1 score: {}%".format(f1*100))
        return accuracy, precision, f1

    def plot_accuracy(self, history):
        plt.plot(history.history["accuracy"], label="acc")
        plt.plot(history.history["val_accuracy"], label="val_acc")
        plt.legend()
        plt.savefig("accuracy_plot.png")

    def plot_loss(self, history):
        plt.plot(history.history["loss"], label="loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.legend()


    def plot_misclassified(self, labels, x_test, y_test, y_pred):
        misclass = np.where(y_test != y_pred)  # array with all misclassified pictures
        i = np.random.choice(misclass)    # pick random index from the array
        plt.imshow(x_test[i].reshape(28,28), cmap="gray")  # print the corresponding picture
        plt.title(" True label: {} Predicted: {}".format(labels[y_test[i]], labels[y_pred[i]]))
