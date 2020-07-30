#!/usr/bin/env python
# coding: utf-8

# # Section 1 -  x^2 Approximation

# ## 1. Using MLP

# In[1]:


# importing needed modules
import jdc as jdc
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense


# In[2]:


# Generating training, cross-validation and test data

X_train = np.random.uniform(low = -20, high = 20, size = 2000)
Y_train = np.square(X_train)

X_dev = np.random.uniform(low = -3, high = 3, size = 200)
Y_dev = np.square(X_dev)

X_test = np.random.uniform(low = -3, high = 3, size = 100)
X_test.sort()
Y_test = np.square(X_test)


# In[36]:


# Define model

model = Sequential()

model.add(Dense(80, input_dim=1, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1, activation='relu'))


# In[37]:


# Compiling Model
model.compile(loss='mean_squared_error', optimizer='adam')


# In[38]:


# Training the model
history = model.fit(X_train, Y_train, epochs=1500, batch_size=100, validation_data=(X_dev, Y_dev))


# In[39]:


# Make predictions on test set

Y_predict = model.predict(X_test)


# In[40]:


# Plotting the learning curves

fig, ax = plt.subplots()
x = range(1,len(history.history['loss'])+1)
y = range(0,31603,10)
ax.grid(color='white', alpha=0.25)
plt.title("Validation Data")
plt.xlabel("Epoch")
plt.ylabel("Rate(%)")
plt.xticks(x)
plt.yticks(y)
plt.ylim(-5,105)
ax.plot(x,100 * np.array(history.history['val_loss']), label= 'loss')
ax.legend()
plt.show()
fig, ax = plt.subplots()
ax.grid(color='white', alpha=1)
ax.set_axisbelow(True)
plt.title("Training Data")
plt.xlabel("Epoch")
plt.ylabel("Rate(%)")
plt.xticks(x)
plt.yticks(y)
plt.ylim(-5,105)
ax.plot(x,100*np.array(history.history['loss']), label= 'loss')
ax.legend()
plt.show()


# In[41]:


# Plotting and comparing

plt.title("Comparing with real")
plt.plot(X_test,Y_test, label="Actual")
plt.plot(X_test,Y_predict, label="Predicted")
plt.legend()
plt.show()


# ## 2. Using RBF Net
# ### Defining helper functions
#     - Defining the RBF Function used in the network, In this example it is the Gaussian Kernel

# In[13]:


def Kernel(x, c, s):
    return np.exp(-1 / (2 * np.square(s))) * np.square((x-c.T))


#     - Using K means to compute cluster centers

# In[14]:


def Kmeans(X, k):

    # Initializing clusters randomly from input data
    clusters = np.random.choice(np.squeeze(X), size = k)
    previous_clusters = clusters.copy()
    standard_deviations = np.zeros(k)
    converged = False

    while not converged:

        # Computing distances for each point to each clustter, and adding an extra dimension to X and clusters
        distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))

        # Finding the cluster that's closest to each point
        closest_cluster = np.argmin(distances, axis = 1)

        # Updating clusters by taking the mean of all of the points assigned to that cluster
        for i in range(k):
            points_for_each_cluster = X[closest_cluster == i]
            if len(points_for_each_cluster) > 0:
                clusters[i] = np.mean(points_for_each_cluster, axis=0)

        # converge if clusters haven't moved more than e^-6
        converged = np.linalg.norm(clusters - previous_clusters) < 1e-6
        previous_clusters = clusters.copy()

    distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))
    closest_cluster = np.argmin(distances, axis=1)

    empty_clusters = []
    for i in range(k):
        points_for_each_cluster = X[closest_cluster == i]
        if len(points_for_each_cluster) < 2:
            # keep track of clusters with no points or 1 point
            empty_clusters.append(i)
            continue
        else:
            standard_deviations[i] = np.std(X[closest_cluster == i])

    # if there are clusters with 0 or 1 points, take the mean std of the other clusters
    if len(empty_clusters) > 0:
        average_points = []
        for i in range(k):
            if i not in empty_clusters:
                average_points.append(X[closest_cluster == i])
        average_points = np.concatenate(average_points).ravel()
        standard_deviations[empty_clusters] = np.mean(np.std(average_points))

    return clusters, standard_deviations


# ### Definning RBF Net Class

# In[15]:


class RBF_NN(object):

    def __init__(self, k = 2, kernel = Kernel):

        self.k = k
        self.kernel = kernel

        self.w = np.random.randn(k,1)
        self.b = np.random.randn(1)

    def fit(self, X, y):

        # use a fixed std
        self.centers, _ = Kmeans(X, self.k)
        self.centers = self.centers.reshape((self.centers.shape[0],1))
        self.stds = max([np.abs(c1 - c2) for c1 in self.centers for c2 in self.centers])

        # Computing weights using matrix inverse
        A = self.kernel(X, self.centers, self.stds)
        Ai = np.linalg.pinv(A)
        self.w = np.dot(Ai, Y_train)

    def predict(self, X):
        Y_predict = []
        A = self.kernel(X, self.centers, self.stds)
        F = np.dot(A, self.w)
        return F


# In[73]:


# Generating training and test data

X_train = np.random.uniform(low = -3, high = 3, size = (2000, 1))

Y_train = np.power(X_train, 2)
# print(X_train)
# print(Y_train)

X_test = np.random.uniform(low = -1500, high = 1500, size = (100, 1))
X_test.sort(axis = 0)

Y_test = np.power(X_test, 2)


# In[74]:


rbf_nn = RBF_NN(k=200)


# In[75]:


rbf_nn.fit(X_train, Y_train)


# In[76]:


Y_predict = rbf_nn.predict(X_test)


# In[77]:


# Plotting and comparing

plt.title("Comparing with real")
plt.plot(X_test, Y_test, label="Actual")
plt.plot(X_test, Y_predict, label="Predicted")
plt.legend()
plt.show()


# ##### 3. Analysis

#     - It is shown that when using RBF networks and approximating functions using gaussians, we can get a lot better results because of Aprroximation theorem which states that every functions can be approximated bus using N number of Gaussians. Also I could use a Inverse-Matrix approach for computing the weights, and It was a lot faster and computationaly efficient.

# # Section 2 - SOFM

# ## 1. Unsupervised Learning

#     - Unsupervised Learning, learns the hidden and intrinsic patterns inside a data set, Due to this nature, Unsupervised Learning is usually used for Clustering uses.

# ## 2. Equations

# ### 1st equation
#     - This equation is the gaussian kernel used for calculating distances between two nodes, and is used to determine neighborhood size in self organization proess.
#
# ### 2nd equation
#     - delta(w) is the value that should be should be added to the existing weights of the network and is computed during every itertion.
#     - eta(t) is value of learning rate computed over timestamp (epoch) t (we use learning rate decay with exponential function).
#     - T is the value of topological neighborhood computed for distance between vector and winning node and its neighbors (within the radius), this value is calculated using equation 1.

# ## 3. Dimensionality Reduction

# In[5]:
