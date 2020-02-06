import csv
import os

import matplotlib.pyplot as plt
import argparse
EPS = 0.0002
LR = 0.2
theta0, theta1 = 0.0, 0.0
stat = False


def estimatePrice(x):
    return theta0 + theta1 * x


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


class turn_stat(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        global stat
        stat = True


def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - float(actual[i])
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return mean_error / mean_error


def simple_predict(test):
    predictions = list()
    for row in test:
        predictions.append(theta0 + theta1 * float(row[0]))
    return predictions


def scale(data):
    max0 = max(data)[0]
    min0 = min(data)[0]
    scaleddata = []
    for elem in data:
        new = [(elem[0] / (max0 - min0)), elem[1]]
        scaleddata.append(new)
    return scaleddata


def evaluete_thetas_cost(data):
    sigma, temp_theta0, temp_theta1, m = 0.0, 0.0, 0.0, len(data)

    for row in data:
        sigma += estimatePrice(row[0]) - row[1]
        temp_theta0 += sigma
        temp_theta1 += sigma * row[0]

    return theta0 - LR / m * (temp_theta0 / m),\
           theta1 - LR / m * (temp_theta1 / m),\
           1 / m * (temp_theta0 * temp_theta0)


def gradient(data):
    global theta0, theta1
    cost, cycles, previouscost = 0, 0, 1

    while abs(cost - previouscost) > EPS:
        previouscost = cost
        theta0, theta1, cost = evaluete_thetas_cost(data)
        cycles += 1

    if stat:
        print("Cycles: ", cycles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get arguments')
    parser.add_argument('-f', action='store', dest='f', help='Input file name')
    parser.add_argument('-s', action=turn_stat, nargs=0)
    args = parser.parse_args()

    if not args.f or not os.path.exists(args.f):
        print("usage: training.py [-h] [-f] \n positional arguments:  -f     Input file name \n                      "
              "  -s     Dispay info about training\n optional arguments:    -h     for help")
        exit()

    data = getData(args.f)
    max0, min0 = max(data)[0], min(data)[0]

    gradient(scale(data))

    theta1 /= max0 - min0                                       # unscale thetha after eval

    print(theta0, theta1)
    f = open('scales.csv', 'w')
    f.write(str(theta0) + ", " + str(theta1))
    f.close()

    if stat:
        test_set = list()
        for row in data:
            row_copy = list(row)                                # evaluate RMSE
            row_copy[-1] = None
            test_set.append(row_copy)
        predicted = simple_predict(test_set)
        actual = [row[-1] for row in data]
        print('RMSE: %.3f' % (rmse_metric(actual, predicted)))

        fig, ax = plt.subplots()
        ax.scatter([int(row[0]) for row in data], [int(row[1]) for row in data])
        plt.plot([float(row[0]) for row in data], predicted, "r")
        ax.set_xlabel('km')
        ax.set_ylabel('price')                                  # plot building
        plt.show()
