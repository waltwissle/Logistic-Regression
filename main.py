import numpy as np
import matplotlib.pyplot as plt
import time
import costFunction, gradient, dbound, predict, sigmoid
import scipy.optimize as op


## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label (0 or 1).

data = np.loadtxt('data.txt', delimiter = ',')

X = data[:,0:2]; y = data[:,2]; Y = y.reshape(100,1)

## ==================== Plotting: Visualize the data ====================
#  We start by first plotting the data to understand the data.
pos = y == 1; neg = y== 0
print('--' * 44)
print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')
print('--' * 44)

plt.figure()
plt.plot(X[pos,0],X[pos,1], 'k+',linewidth=12, markersize=7, label = 'Admitted' ); 
plt.plot(X[neg,0],X[neg,1], 'ko', linewidth=2, markersize=6.5, mfc = 'r', label = 'Not Admitted'); 
plt.grid(linestyle ='--')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.title('Exam scores')
plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.show()

time.sleep(2) # pause for 2 secs

m, n = np.shape(X)
phi = np.concatenate((np.ones((m,1)), X), axis  = 1)

# initial theta          
initial_theta = np.zeros((n + 1, 1))

#compute and display initial  cost and gradient
J = costFunction.costFunction(initial_theta, phi, Y)
grad = gradient.gradient(initial_theta, phi, Y)

print( f'Cost at initial theta (zeros): {round(J,2)}');
print('Expected cost (approx): 0.693\n');
print('Gradient at initial theta (zeros): \n');
print(f' {grad}');
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n');


# Compute and display cost and gradient with non-zero theta
test_theta = np.array([[-24, 0.2, 0.2]]).T;

Ja = costFunction.costFunction(test_theta, phi, Y)
grada = gradient.gradient(test_theta, phi, Y)
print(f'\nCost at test theta: \n {Ja}\n');
print('Expected cost (approx): 0.218\n');
print('Gradient at test theta: \n');
print(f' {round(grada[0,0], 2)} \n {round(grada[0,1], 2)} \n {round(grada[0,2],2)}\n\n');
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n');


## ============= Part 3: Optimizing using op.minimize  =============

print('==' *30)
print('Optimization using minimize from the scipy.optimize module')
print('==' *30)

m , n = phi.shape;
initial_theta = np.zeros(n);
Result = op.minimize(fun = costFunction.costFunction, x0 = initial_theta, args = (phi, Y), method = 'TNC',jac = gradient.gradient)


print(f'\nCost at initial theta found by fmin_tnc:\n {round(Result.fun,2)}');
print('\nExpected cost (approx):\n 0.203\n');
print(f'theta: \n {round(Result.x[0],2)}\n {round(Result.x[1],2)}\n {round(Result.x[2],2)}\n');
print('Expected theta (approx):');
print(' -25.161\n 0.206\n 0.201\n');

# Calculate the decision boundary
bndl = dbound.dbound(Result.x,X)

print('PLotting the Decision boundary...\n')

plt.figure()
plt.plot(X[pos,0],X[pos,1], 'k+',linewidth=12, markersize=7, label = 'Admitted' ); 
plt.plot(X[neg,0],X[neg,1], 'ko', linewidth=2, markersize=6.5, mfc = 'r', label = 'Not Admitted'); 
plt.plot(X[:,0],bndl, 'b-', linewidth= 0.5, label = 'Decision Boundary')
plt.grid(linestyle ='--')
plt.xlim(30,100)
plt.ylim(30,100)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.title('Exam scores and Decision boundary')
plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.show()


print('==' *30)
print('Determine the accuracy of the classifier')
print('==' *30)

p = predict.predict(Result.x, phi)

print(f'Train Accuracy: {(np.mean(p==Y) * 100)}%')


print('\n\nTest the Classifier: Determine if a student with Test 1 & 2 scores = 45 and 85 respectively')

prob = sigmoid.sigmoid([1, 45 ,85] @ Result.x);
print(f'\n\nFor a student with scores 45 and 85, we predict an admission probability of:\n{round(prob * 100,2)}%');






