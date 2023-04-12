# import the Iris dataset from scikit-learn
from sklearn.datasets import load_iris
# import our plotting module
import matplotlib.pyplot as plt

iris = load_iris()

iris_X, iris_y =  iris.data, iris.target

print(iris.target_names)
print(iris.feature_names)

# for labelling: {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
label_dict = {i: k for i, k in enumerate(iris.target_names)}

def plot(X, y, title, x_label, y_label):
    ax = plt.subplot(111)
    for label,marker,color in zip(
    range(3),('^', 's', 'o'),('blue', 'red', 'green')):

        plt.scatter(x=X[:,0].real[y == label], y=X[:,1].real[y == label],
            color=color,
            alpha=0.5,
            label=label_dict[label]
            )

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title(title)

# plot(iris_X, iris_y, "Original Iris Data", "sepal length (cm)", "sepal width (cm)")
# plt.show()


# Calculate a PCA manually

# import numpy
import numpy as np

# calculate the mean vector
mean_vector = iris_X.mean(axis=0)
print(mean_vector)

print(iris_X.shape)
print((iris_X).T.shape)

# calculate the covariance matrix
# 计算协方差矩阵
cov_mat = np.cov((iris_X).T)
print(cov_mat.shape)
print(cov_mat)
print("-----------------")

# calculate the eigenvectors and eigenvalues of our covariance matrix of the iris dataset
# 计算特征向量和特征值
eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
print(eig_vec_cov)
print(eig_val_cov)
print('-----------------')

# Print the eigen vectors and corresponding eigenvalues
# in order of descending eigenvalues
for i in range(len(eig_val_cov)):
    eigvec_cov = eig_vec_cov[:,i]
    print('Eigenvector {}: \n{}'.format(i+1, eigvec_cov))
    print('Eigenvalue {} from covariance matrix: {}'.format(i+1, eig_val_cov[i]))
    print(30 * '-')

# the percentages of the variance captured by each eigenvalue
# is equal to the eigenvalue of that components divided by
# the sum of all eigen values
explained_variance_ratio = eig_val_cov/eig_val_cov.sum()
print(explained_variance_ratio)

print("=======-======00000")

# Scree Plot
# plt.plot(np.cumsum(explained_variance_ratio))
# plt.title('Scree Plot')
# plt.xlabel('Principal Component (k)')
# plt.ylabel('% of Variance Explained <= k')
# plt.show()

# store the top two eigenvectors in a variable
top_2_eigenvectors = eig_vec_cov[:,:2].T

# show the transpose so that each row is a principal component, we have two rows == two components
print(top_2_eigenvectors)
print(top_2_eigenvectors.shape)
print(top_2_eigenvectors.T.shape)
print(iris_X.shape)

print(np.dot(iris_X, top_2_eigenvectors.T)[:5,])


print("============================")
# scikit-learn's version of PCA
from sklearn.decomposition import PCA

# Like any other sklearn module, we first instantiate the class
pca = PCA(n_components=2)

# fit the PCA to our data
pca.fit(iris_X)

#plot(iris_X, iris_y, "Original Iris Data", "sepal length (cm)", "sepal width (cm)")

plot(pca.transform(iris_X), iris_y, "Iris: Data projected onto first two PCA components", "PCA1", "PCA2")
plt.show()
















