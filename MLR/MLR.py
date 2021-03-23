import numpy as np


def cost_function(X, Y, B):
 m = len(Y)
 cost = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
 return cost

def gradient_descent(X, Y, B, learning_rate, iterations):
 m = len(Y)
 
 for iteration in range(iterations):
    current_value = X.dot(B)
    loss = current_value - Y
    gradient = X.T.dot(loss) / m
    B = B - learning_rate * gradient
    cost = cost_function(X, Y, B)
 
 return B

def prediction(x_test, MR):
    return x_test.dot(MR)