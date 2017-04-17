import csv
import math
import numpy
import matplotlib.pyplot as plt
# check following link on how to use cvxopt: 
# https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
from cvxopt import matrix
from cvxopt import solvers

def prep_graph(x, y):
	x1 = []
	x2 = []
	for i in range(len(x)):
		if y[i] == 1:
			x1.append(x[i, :])
		else:
			x2.append(x[i, :])
	x1 = numpy.array(x1)
	x2 = numpy.array(x2)
	plt.plot(x1[:,0], x1[:,1], 'r+')
	plt.plot(x2[:,0], x2[:,1], 'b+')

def part1_a(x, y):
	prep_graph(x, y)
	plt.show()

def part1_bcde(x, y):
	# QP to solve primal form for SVM -- convert to cxvopt matrices
	P = numpy.eye(3)
	P[2][2] = 0
	P = matrix(P, tc='d')
	q = matrix(numpy.zeros(3), tc='d')
	G = []
	for i in range(len(y)):
		G.append([-1 * y[i] * x[i, 0], -1 * y[i] * x[i, 1], -1 * y[i]])
	G = matrix(numpy.array(G), tc='d')
	h = matrix(-1 * numpy.ones(len(y)), tc='d')
	sol = solvers.qp(P, q, G, h)
	
	# part b
	theta = numpy.array([sol['x'][0], sol['x'][1]])
	theta_0 = sol['x'][2]
	opt_value = sol['primal objective']
	print "is optimal status : " + str(sol["status"])
	print "optimal theta_0 is : " + str(theta_0)
	print "optimal theta is : " + str(theta)
	print "optimal objective function value is : " + str(opt_value)

	# part c
	prep_graph(x, y)	# original points
	# find line 
	min_x1 = min(x[:, 0])
	min_x2 = (0 - theta[0] * min_x1 - theta_0) / float(theta[1])
	max_x1 = max(x[:, 0])
	max_x2 = (0 - theta[0] * max_x1 - theta_0) / float(theta[1])
	plt.plot(numpy.array([min_x1, max_x1]), numpy.array([min_x2, max_x2]))
	# compute support vectors
	dist_matrix = y * (numpy.dot(x, theta) + theta_0)
	min_dist = min(dist_matrix)
	support_vectors_x1 = []
	support_vectors_x2 = []
	for i in range(len(y)):
		if (dist_matrix[i] <= min_dist + 1.0 / 1000000):	# to prevent floating number error
			support_vectors_x1.append(x[i, 0])
			support_vectors_x2.append(x[i, 1])
	plt.plot(support_vectors_x1, support_vectors_x2, 'gx')
	plt.show()

	# part d
	indices = []
	values = []
	for i in range(len(sol['z'])):
		if sol['z'][i] > 1.0/1000000:
			indices.append(i)
			values.append(sol['z'][i])
	print "number of non-zero entries : " + str(len(indices))
	print "indices of non-zero entries (starting from 0 index) : " + str(indices)
	print "and corresponding values : " + str(values)
	print "dual optimal objective function value is : " + str(sol['dual objective'])

	# part e
	first_sum = [0, 0]
	for i in range(len(indices)):
		first_sum += values[i] * y[indices[i]] * x[indices[i], :]
	print "the first sum is : " + str(first_sum)

	j = numpy.random.randint(len(indices))
	second_sum = 0
	for i in range(len(y)):
		second_sum += sol['z'][i] * y[i] * numpy.dot(x[i, :], numpy.transpose(x[indices[j], :]))
	second_sum = y[indices[j]] - second_sum
	print "the second sum is : " + str(second_sum)

# def part1_de(x, y):
	# length = len(y)
	# P = []
	# A = []
	# for i in range(length):
	# 	P.append([y[i] * x[i, 0], y[i] * x[i, 1], y[i]])
	# 	A.append([y[i]])
	# P = numpy.dot(P, numpy.transpose(P))
	# P = matrix(P, tc='d')
	# q = matrix(-1 * numpy.zeros(length), tc='d')
	# G = matrix(-1 * numpy.identity(length), tc='d')
	# h = matrix(numpy.zeros(length), tc='d')
	# A = matrix(A, tc='d')
	# b = matrix(0, tc='d')
	# sol = solvers.qp(P, q, G, h, A, b)

def part_2(x, y):
	x1 = []
	x2 = []
	for i in range(len(x)):
		if y[i] == 1:
			x1.append(x[i, :])
		else:
			x2.append(x[i, :])
	x1 = numpy.array(x1)
	x2 = numpy.array(x2)
	plt.plot(x1[:,0], x1[:,1], 'rx')
	plt.plot(x2[:,0], x2[:,1], 'b+')
	plt.show()

# main
# reader = csv.reader(open("iris1.csv", "rb"), delimiter=",")
reader = csv.reader(open("iris2.csv", "rb"), delimiter=",")
data = numpy.array(list(reader))
y = data[:, len(data[0]) - 1]
y = y.astype(int)
x = numpy.delete(data, len(data[0])-1, axis=1)
x = x.astype(float)

# part1_a(x, y)
# part1_bcde(x, y)
part_2(x, y)