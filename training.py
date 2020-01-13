import csv
import matplotlib.pyplot as plt
from math import sqrt


def getData(filename):
    global data
    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        next(csvReader)
        for row in csvReader:
            row0 = float(row[0])
            row1 = float(row[1])
            data.add((row0, row1))


def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - float(actual[i])
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)

def scale(maxX, minX, dataset):
    scaleddata = []
    for data in dataset:
        new = [(data[0] / (maxX[0] - minX[0])), data[1]]
        scaleddata.append(new)
    return (scaleddata)

def evaluate_algorithm(dataset, algorithm):
    test_set = list()
    for row in dataset:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    predicted = algorithm(dataset, test_set)
    actual = [row[-1] for row in dataset]
    rmse = rmse_metric(actual, predicted)
    return rmse


def mean(values):
    return sum(values) / float(len(values))


def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar


def variance(values, mean):
    return sum([(x-mean)**2 for x in values])


def coefficients(dataset):
    x = [int(row[0]) for row in dataset]
    y = [int(row[1]) for row in dataset]
    x_mean, y_mean = mean(x), mean(y)
    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1 * x_mean
    return [b0, b1]


def simple_linear_regression(train, test):
    global predictions, tt0, tt1
    for row in test:
        yhat = tt0 + tt1 * float(row[0])
        predictions.append(yhat)
    return predictions


def estimatePrice(x):
    global tt0, tt1
    return tt0 + tt1 * x

def gradient(scaled):
    global tt0, tt1, maxX, minX
    costfunction = 0
    previouscost = -999999
    # for i in range(10):
    i = 0
    while abs(costfunction - previouscost) > 0.0003:
        previouscost = costfunction
        sigma, tmp0, tmp1 = 0.0, 0.0, 0.0
        for row in scaled:
            sigma += estimatePrice(row[0]) - row[1]
            tmp0 += sigma
            tmp1 += sigma * row[0]
        tt0 = tt0 - learningRate * (1 / m) * (tmp0 / m)
        tt1 = tt1 - learningRate * (1 / m) * (tmp1 / m)
        costfunction = 1 / m * (tmp0 * tmp0)
        i += 1
    print(i)
    tt1 /= maxX[0] - minX[0]

data = set()
getData('data.csv')
m = float(len(data))
tt0, tt1 = 0.0, 0.0
learningRate = 3
predictions = list()

# bonus part


maxX = max(data)
minX = min(data)
scaled = scale(maxX, minX, data)
gradient(scaled)
rmse = evaluate_algorithm(data, simple_linear_regression)
print('Precision: %.3f' % (rmse))
print(tt0, tt1)
f = open('scales.csv', 'w')
f.write(str(tt0) + ", " + str(tt1))
f.close()
fig, ax = plt.subplots()

ax.scatter([int(row[0]) for row in data], [int(row[1]) for row in data])
plt.plot([float(row[0]) for row in data], predictions, "r")

ax.set_xlabel('km')
ax.set_ylabel('price')
plt.show()
# gradient()
# estimatePrice(mileage) = θ0 + (θ1 ∗ mileage)
#
#                             m−1
# tmpθ0 = learningRate ∗ 1/m   E (estimatePrice(mileage[i]) − price[i])
#                             i=0
#
#                             m−1
# tmpθ1 = learningRate ∗ 1/m   E (estimatePrice(mileage[i]) − price[i]) ∗ mileage[i]
#                             i=0
#