# John Shapiro
# Machine Learning Spring 2020
# Homework 1

import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([35., 38., 31., 20., 22., 25., 17., 60., 8., 60.])
y_data = 2*x_data+50+5*np.random.random()

bias = np.arange(0,100,1) #bias
weight = np.arange(-5, 5,0.1) #weight
Z = np.zeros((len(bias),len(weight)))

for i in range(len(bias)):
    for j in range(len(weight)):
        b = bias[i]
        w = weight[j]
        Z[j][i] = 0
        for n in range(len(x_data)):
            Z[j][i] = Z[j][i] + (w * x_data[n] + b - y_data[n]) ** 2
        Z[j][i] = Z[j][i]/len(x_data)

plt.xlim(0,100)
plt.ylim(-5,5)
plt.plot(52, 2,"X", color = "yellow", markersize = 10) # plot the target
plt.contourf(bias,weight,Z, 50, alpha =0.5, cmap= plt.get_cmap('jet'))

b = 1 # initial b
w = 1 # initial w
lr = 0.00015 #learning rate
iteration = 100000
b_history = [b]
w_history = [w]

iterations = 0
target_gradient = 0.0001

for i in range(iteration):
    iterations += 1
    b_grad = 0.0
    w_grad = 0.0
    for n in range(len(x_data)):
        b_grad = b_grad + (b + w*x_data[n] - y_data[n]) * 1.0
        w_grad = w_grad + (b + w*x_data[n] - y_data[n]) * x_data[n]
    # stop iterations when target gradient is reached
    if abs(b_grad) < target_gradient and abs(w_grad) < target_gradient:
        print(f"Reached target gradient of less than 0.0001 in {iterations} iterations.")
        break
    b -= lr * b_grad
    w -= lr * w_grad
    b_history.append(b)
    w_history.append(w)

# calculate h(x) using b and w
hx_data = w * x_data + b

# sum up the squares of h(x) - y and divide by two
sum_of_squares = 0
for i in range(len(y_data)):
    sum_of_squares += (hx_data[i] - y_data[i]) ** 2
loss = sum_of_squares / 2.0

print(f"Calculated loss J(w) = {loss}")


plt.plot(b_history, w_history, 'o-', ms=.5, lw=1.5,color='black')
plt.plot(52, 2,"X", color = "yellow", markersize = 10)
plt.title(f'LR = ' + '%f'%lr + f'  Iterations = {iterations}')
plt.show()