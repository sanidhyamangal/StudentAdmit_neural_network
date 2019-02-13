import numpy as np 
from data_prep import features, target, features_test, target_test

# sigmoid function to perform activation 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid prime for derivative 
def _sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Seeding to make debbuging easy 
np.random.seed(42)

n_records, n_features = features.shape

last_loss = None

# Initialize weights
weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

# Neural Network hyperparameters 
num_epochs = 1000
learningrate = 0.5


for e in range(num_epochs):
    del_w = np.zeros(n_features)

    for x, y in zip(features.values, target):

        # out before activation 
        output = np.dot(x, weights)

        # activating the output 
        nn_output = sigmoid(output)

        # error calculation 
        error = (y - nn_output)

        # error term for the backpropogation 
        error_term = error * _sigmoid(output)

        # calculate the change in weights 
        del_w += error_term * x

    # Update weights using learning rate and del_w
    weights += learningrate * del_w / n_records

    # Printing the training step

    if e  % (num_epochs / 10) == 0:
        
        # output of nn
        out = sigmoid(np.dot(features, weights))

        # loss
        loss = np.mean((target - out) ** 2)

        # check wether the the loss is increasing or not 
        if last_loss and last_loss < loss:
            print("Train loss", loss, "WARNING - Loss increasing")
        else:
            print("Train loss", loss)
        
        last_loss = loss # to c

# calculate accuracy on test set 
test_out = sigmoid(np.dot(features_test, weights))

prediction = test_out > 0.5

accuracy = np.mean(prediction == target_test)

print("Accuracy", accuracy)





