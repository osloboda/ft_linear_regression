import csv
import matplotlib.pyplot as plt
from math import sqrt
LR = 0.2
EPS = 0.0002


def getData(filename):
    data = set()
    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        next(csvReader)
        for row in csvReader:
            row0 = float(row[0])
            row1 = float(row[1])
            data.add((row0, row1))
    return data


def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - float(actual[i])
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)


def simple_linear_regression(test, theta0, theta1):
    predictions = list()
    for row in test:
        yhat = theta0 + theta1 * float(row[0])
        predictions.append(yhat)
    return predictions


def estimatePrice(x, theta0, theta1):
    return theta0 + theta1 * x


def gradient(data, theta0, theta1, max0, min0):
    costfunction, cycles = 0, 0
    previouscost = 10 ** 6
    scaleddata = []
    m = float(len(data))
    for elem in data:
        new = [(elem[0] / (max0 - min0)), elem[1]]
        scaleddata.append(new)
    while abs(costfunction - previouscost) > EPS:
        previouscost = costfunction
        sigma, tmp0, tmp1 = 0.0, 0.0, 0.0
        for row in scaleddata:
            sigma += estimatePrice(row[0], theta0, theta1) - row[1]
            tmp0 += sigma
            tmp1 += sigma * row[0]
        theta0 = theta0 - LR / m * (tmp0 / m)
        theta1 = theta1 - LR / m * (tmp1 / m)
        costfunction = 1 / m * (tmp0 * tmp0)
        cycles += 1
    print("Cycles: ", cycles)
    theta1 /= max0 - min0
    return theta0, theta1


data = getData('data.csv')
theta0, theta1 = 0.0, 0.0

# bonus part
max0 = max(data)[0]
min0 = min(data)[0]

(theta0, theta1) = gradient(data, theta0, theta1, max0, min0)

test_set = list()
for row in data:
    row_copy = list(row)
    row_copy[-1] = None
    test_set.append(row_copy)
predicted = simple_linear_regression(test_set, theta0, theta1)
actual = [row[-1] for row in data]

print('Precision: %.3f' % (rmse_metric(actual, predicted)))
print(theta0, theta1)
f = open('scales.csv', 'w')
f.write(str(theta0) + ", " + str(theta1))
f.close()
fig, ax = plt.subplots()

ax.scatter([int(row[0]) for row in data], [int(row[1]) for row in data])
plt.plot([float(row[0]) for row in data], predicted, "r")

ax.set_xlabel('km')
ax.set_ylabel('price')
plt.show()
