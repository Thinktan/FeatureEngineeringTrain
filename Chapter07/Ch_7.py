# import numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, datasets, metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

# create numpy array from csv
images = np.genfromtxt('../data/mnist_train.csv', delimiter=',')

images_X, images_y = images[:,1:], images[:,0]
# plt.imshow(images_X[0].reshape(28, 28), cmap=plt.cm.gray_r)
# plt.show()

# scale images_X to be beteen 0 and 1
images_X = images_X / 255.

# make pixels binary (either white or black)
images_X = (images_X > 0.5).astype(float)

# instantiate our BernoulliRBM
# we set a random_state to initialize our weights and biases to the same starting point
# verbose is set to True to see the fitting period
# n_iter is the number of back and forth passes
# n_components (like PCA and LDA) represent the number of features to create
# n_components can be any integer, less than , equal to, or greater than the original number of features
rbm = BernoulliRBM(random_state=0, verbose=True, n_iter=20, n_components=100)

rbm.fit(images_X)

image_new_features = rbm.transform(images_X[:1]).reshape(100,)

print(image_new_features)

# get the most represented features
top_features = image_new_features.argsort()[-20:][::-1]

print(top_features)
print(image_new_features[top_features])

# plot the RBM components (representations of the new feature sets) for the most represented features
plt.figure(figsize=(25, 25))
for i, comp in enumerate(top_features):
    plt.subplot(5, 4, i + 1)
    plt.imshow(rbm.components_[comp].reshape((28, 28)), cmap=plt.cm.gray_r)
    plt.title("Component {}, feature value: {}".format(comp, round(image_new_features[comp], 2)), fontsize=20)

# plt.suptitle('Top 20 components extracted by RBM for first digit', fontsize=30)
# plt.show()






