import csv
import math
import numpy
import matplotlib.pyplot as plt
# check following link on how to use cvxopt: 
# https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
from cvxopt import matrix
from cvxopt import solvers

def get_min_gamma(x, y, theta):
	min_gamma = float("inf")
	for i in range(len(x)):
		min_gamma = min(min_gamma, y[i] * numpy.dot(theta, x[i, :]))
	return min_gamma

def perceptron(x, y, theta_0, s_index=0):
	theta = theta_0
	index = s_index
	counter = 0
	tot_updates = 0
	max_iterations = 1000000
	while counter < len(x):
		# check if classified correctly
		gamma = y[index] * numpy.dot(theta, x[index, :])
		if gamma <= 0:
			counter = 0
			tot_updates += 1
			theta += numpy.dot(int(y[index]), x[index, :])
			if (tot_updates > max_iterations):
				raise Exception('Maximum number of iterations exceed!')
		else:
			counter += 1
		index = (index + 1) % len(x)
	# normalise theta before return
	theta /= numpy.linalg.norm(theta)
	min_gamma = get_min_gamma(x, y, theta)
	return [theta, min_gamma, tot_updates]

def sample_unit_circle(npoints=1, ndim=2):
	vec = numpy.random.randn(ndim, npoints)
	vec = numpy.array([vec[0][0], vec[1][0]])
	return vec / numpy.linalg.norm(vec)

def compute_bound(theta_opt, theta_0, gamma):
	a = numpy.dot(theta_opt, theta_0)
	first_bound = -1.0 * a / gamma
	second_bound = (1 - 2*a*gamma + 
			math.sqrt( math.pow(2*a*gamma-1, 2) - 4*gamma*gamma * (a*a - 1) )
		) / (2*gamma*gamma)
	return max(first_bound, second_bound)


def part1(x, y):
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
	plt.show()


def part2(x, y):
	# QP to solve primal form for SVM
	# convert to cxvopt matrices
	P = matrix(numpy.eye(2), tc='d')
	q = matrix(numpy.zeros(2), tc='d')
	G = []
	for i in range(len(y)):
		G.append([-1 * y[i] * x[i, 0], -1 * y[i] * x[i, 1]])
	G = matrix(numpy.array(G), tc='d')
	h = matrix(-1 * numpy.ones(len(y)), tc='d')
	sol = solvers.qp(P,q,G,h)

	theta = numpy.array([sol['x'][0], sol['x'][1]])
	theta /= numpy.linalg.norm(theta)
	min_gamma = get_min_gamma(x, y, theta)
	print "is optimal status : " + str(sol["status"])
	print "optimal theta is : " + str(theta)
	print "corrsponding minimum gamma is : " + str(min_gamma)


def part3(x, y):
	# part a
	print "\nsubtask 3 part (a):"
	result = perceptron(x, y, numpy.zeros(len(x[0])))
	print "number of iterations: " + str(result[2])
	print "converged solution theta: " + str(result[0])
	print "corresponding minimum gamma: " + str(result[1])

	# part b
	print "\nsubtask 3 part (b):"
	for i in range(10):
		theta_0 = numpy.array([numpy.random.rand(), numpy.random.rand()])
		print "starting point: " + str(theta_0)
		print perceptron(x, y, theta_0)

	# part d
	print "\nsubtask 3 part (d):"
	tot_iteration = 0.0
	tot_gamma = 0.0
	# variables to assist plotting
	counter = 0
	upper_bound_lst = []
	iteration_lst = []
	for i in range(10000):
		theta_0 = sample_unit_circle()
		result = perceptron(x, y, theta_0)
		tot_iteration += result[2]
		tot_gamma += result[1]
		if counter <= 100:
			counter += 1
			upper_bound = compute_bound(result[0], theta_0, result[1])
			upper_bound_lst.append(upper_bound)
			iteration_lst.append(result[2])

	avg_iteration = tot_iteration / 10000
	avg_gamma = tot_gamma / 10000
	print("average number of iteration is :" + str(avg_iteration))
	print("average gamma is :" + str(avg_gamma))

	plt.plot(upper_bound_lst, iteration_lst, '+')
	# plt.plot(numpy.array(upper_bound_lst) - numpy.array(iteration_lst), '+')
	plt.show()


def part4(x, y):
	y[0] = -1
	y[2] = 1
	# plot 
	part1(x, y)
	# run step 2 (SVM)
	part2(x, y)
	# perceptron
	perceptron(x, y, numpy.zeros(len(x[0])))


# main
reader = csv.reader(open("Problem1.csv", "rb"), delimiter=",")
data = numpy.array(list(reader))
y = data[:, len(data[0]) - 1]
y = y.astype(int)
x = numpy.delete(data, len(data[0])-1, axis=1)
x = x.astype(float)

# part1(x, y)
# part2(x, y)
# part3(x, y)
part4(x, y)