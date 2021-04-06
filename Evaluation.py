import tensorflow as tf
from CNNModel import CNN
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

class Evaluation:
    def __init__(self, input_shape, nb_classes):
        super(Evaluation, self).__init__()
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.CNN = CNN(input_shape=input_shape, nb_classes=nb_classes)

    def cross_validation(self, n_splits, dataX, dataY, with_generator):
        scores = []
        histories = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        for train_id, test_id in skf.split(dataX, dataY):
          print(dataX[train_id])
          model = self.CNN.build()
          x_train, y_train, x_test, y_test = dataX[train_id], dataY[train_id], dataX[test_id], dataY[test_id]
          history = model.train(model, with_generator=with_generator,
                                x_train=x_train, y_train=y_train,
                                x_test=x_test, y_test=y_test,
                                EPOCHS=10, BATCH_SIZE=32)

          result = model.evaluate(x_test, y_test, verbose=True)
          scores.append(result)
          histories.append(history)

        return scores, histories













