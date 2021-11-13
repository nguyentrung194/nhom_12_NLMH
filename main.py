# Logistic Regression on Diabetes Dataset
from random import seed
from random import randrange
from csv import reader
from math import exp
import random

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())


# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def cross_validation_split2(dataset, test_rate, val_rate):
  random.shuffle(dataset)
  size=len(dataset)
  train_size=size - round(size*(test_rate + val_rate))
  train = dataset[0:train_size]
  test_size =train_size + int(size*test_rate)
  test = dataset[train_size:test_size]
  val = dataset[test_size:-1]
  return train,test,val

def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct = correct + 1
	return correct / float(len(actual)) * 100.0

def accuracy(actual, predicted):
	tn = 0
	fp = 0
	fn = 0
	tp = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			if predicted[i] == 1:
				tp += 1
			if predicted[i] == 0:
				tn += 1
		if actual[i] != predicted[i]:
			if predicted[i] == 1:
				fp += 1
			else:
				fn += 1
	return (tp+tn)/(tp+fp+tn+fn) * 100.0

def recall(actual, predicted):
	tn = 0.0
	fp = 0.0
	fn = 0.0
	tp = 0.0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			if predicted[i] == 1:
				tp += 1
			if predicted[i] == 0:
				tn += 1
		if actual[i] != predicted[i]:
			if predicted[i] == 1:
				fp += 1
			else:
				fn += 1
	print(tn)
	print(fp)
	print(fn)
	print(tp)
	return (tp)/(tp+fn) * 100.0


def precision(actual, predicted):
	tn = 0.0
	fp = 0.0
	fn = 0.0
	tp = 0.0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			if predicted[i] == 1:
				tp += 1
			if predicted[i] == 0:
				tn += 1
		if actual[i] != predicted[i]:
			if predicted[i] == 1:
				fp += 1
			else:
				fn += 1
	return (tp)/(tp+fp) * 100.0

def f1(actual,predicted):
	pre = precision(actual, predicted)
	rec = recall(actual, predicted)
	return (2*pre*rec)/(pre+rec)

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm2(train_set,test_set, algorithm,accuracy_metric, *args):
	scores = list()
	predicted = algorithm(train_set, test_set, *args)
	actual = [row[-1] for row in test_set]
	accuracy = accuracy_metric(actual, predicted)
	scores.append(accuracy)
	return scores
# Make a prediction with coefficients
def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return 1.0 / (1.0 + exp(-yhat))

# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			yhat = predict(row, coef)
			error = row[-1] - yhat
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
	return coef

# Linear Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(train, test, l_rate, n_epoch):
	predictions = list()
	coef = coefficients_sgd(train, l_rate, n_epoch)
	for row in test:
		yhat = predict(row, coef)
		yhat = round(yhat)
		predictions.append(yhat)
	return (predictions)

	# Test the logistic regression algorithm on the diabetes dataset
seed(1)
# load and prepare data
filename = 'diabetes.csv'
dataset = load_csv(filename)

dataset = dataset[1:-1]

train,test,val = cross_validation_split2(dataset,0.15,0.15)

for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)

# normalize
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)

# evaluate algorithm
l_rate = 0.1
n_epoch = 100

scores2 = evaluate_algorithm2(train, test, logistic_regression, accuracy, l_rate, n_epoch)
print('Mean Accuracy: %.3f%%' % (sum(scores2)/float(len(scores2))))

scores3 = evaluate_algorithm2(train, test, logistic_regression, recall, l_rate, n_epoch)
print('Recall: %.3f%%' % (sum(scores3)/float(len(scores3))))

scores4 = evaluate_algorithm2(train, test, logistic_regression,precision, l_rate, n_epoch)
print('Precision: %.3f%%' % (sum(scores4)/float(len(scores4))))

scores5 = evaluate_algorithm2(train,test,logistic_regression, f1, l_rate, n_epoch)
print('F1: %.3f%%' % (sum(scores5)/float(len(scores5))))

# Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome