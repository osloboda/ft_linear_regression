import csv
import numpy as np


def mse_loss(y_true, y_pred):
    # y_true и y_pred являются массивами numpy с одинаковой длиной
    return ((y_true - y_pred) ** 2).mean()


y_true = np.array([1, 0, 0, 1])
y_pred = np.array([0, 0, 0, 0])

print(mse_loss(y_true, y_pred))  # 0.5


def getData(filename):
    x, y = [], []
    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        next(csvReader)
        for row in csvReader:
            temp = []
            for elem in row[1:]:
                    temp.append(elem)
            x.append(temp)
            y.append([float(row[1])])
        return np.array(x), np.array(y)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


training_inputs, training_outputs = getData('dataset_train.csv')
# training_inputs, training_outputs = np.array([[0,0,1],
#                                               [1,1,1],
#                                               [1,0,1],
#                                               [0,1,1]]), np.array([[0,1,1,0]]).T

print(training_inputs, '\n', training_outputs)

np.random.seed(1)

synaptic_weights = 2 * np.random.random((11, 1)) - 4

print(synaptic_weights)

# for i in range(20000):
#     input_layer = training_inputs
#     outputs = sigmoid(np.dot(input_layer, synaptic_weights))
#
#     err = training_outputs - outputs
#     adjustments = np.dot(input_layer.T, err * (outputs * (1 - outputs)))
#     synaptic_weights += adjustments
#
# print(synaptic_weights)
# print(outputs)