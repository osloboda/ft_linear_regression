import csv


data = tuple()
with open('scales.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        data = tuple(row)
print("Enter k50m: ")
x = input()
print("Predicted price: %.f" % (float(data[0]) + float(data[1]) * float(x)))