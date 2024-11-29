import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
# for interactive plotting: https://stackoverflow.com/questions/43189394/interactive-plotting-in-pycharm-debug-console-through-matplotlib
import matplotlib as mpl
mpl.use('TkAgg')  # interactive mode works with this, pick one

def PCA(x, M):
    cov_matrix = np.cov(x, rowvar=False)
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # Indices of eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    # Select the top-M indices
    top_k_indices = sorted_indices[:M]
    # Corresponding eigenvectors
    top_k_eigenvectors = eigenvectors[:, top_k_indices]
    
    return top_k_eigenvectors

######  Load the MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data/255  # Shape: (70000, 784)
y = np.array(mnist.target.astype(int))
X = np.asarray(X)
# Use only the first 60,000 samples for training
X_train = X[:60000]
print("Dataset loaded. Shape of training data:", X_train.shape)

######  Data Processing
# Center the data
mean = np.mean(X_train, axis=0)  # Compute mean of each feature
X_centered = X_train - mean  # Center the data

###### Solution 1-1:  TOP 5 Principle Components ######
print("\n###### Solution 1-1:  TOP 5 Principle Components\n")
M = 5  # Number of components to keep
top_k_eigenvectors = PCA(X_centered, M)
print(f"Top-{M} principal components selected.")

print("Plotting the top 5 principal components...")
fig, axes = plt.subplots(1, M, figsize=(15, 5))
for i in range(M):
    ax = axes[i]
    eigenvector = top_k_eigenvectors[:, i]  # Get the i-th eigenvector
    eigenvector_image = eigenvector.reshape(28, 28)  # Reshape to 28x28 image
    ax.imshow(eigenvector_image, cmap='gray')  # Display as grayscale image
    ax.set_title(f"PC_{i+1}")
    ax.axis('off')  # Hide axes for better visualization
plt.tight_layout()
fig.suptitle("Top 5 Principle Components")
plt.show()

###### Solution 1-2: Selecting one image and plotting values ######
print("\n###### Solution 1-2:  Selecting one image and plotting values\n")
# Randomly select one image from the training set (x_n)
random_idx = np.random.randint(0, X_train.shape[0])  # Random index
x_n = X_centered[random_idx]  # Selected image (centered)

# Project x_n onto the top-5 eigenvectors to get z_n
z_n = top_k_eigenvectors.T @ x_n
print("(a) find z_n")
print(z_n)
# Reconstruct the image using the top-5 principal components
x_tilde_n = top_k_eigenvectors @ z_n  # Reconstructed image

# Plotting Original and reconstructed image
fig, axes = plt.subplots(1, M+2, figsize=(15, 10))

print("(b) Plot")
# Plot original image
axes[0].imshow((x_n+mean).reshape(28, 28), cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

# Plot reconstructed image
axes[1].imshow((x_tilde_n+mean).reshape(28, 28), cmap='gray')
axes[1].set_title("Reconstructed Image")
axes[1].axis('off')

# Plot the contributions of each principal component
for i in range(M):
    ax = axes[i+2]
    # Plot the contribution of each principal component: z_i * b_i
    contribution = z_n[i] * top_k_eigenvectors[:, i].reshape(28, 28)
    ax.imshow(contribution, cmap='gray')
    ax.set_title(f"z_{i} * b_{i} > PC{i+1} ")
    ax.axis('off')

fig.suptitle("Reconstruction using different PC components")
plt.tight_layout()
plt.show()

# (c) Rconstruction error for one sample
reconstruction_error = np.linalg.norm(x_n - x_tilde_n)

input = x_n
target = x_tilde_n
squared_diff = (input - target) ** 2
mean_reconstruction_error = np.mean(squared_diff)
print(f"(c) Reconstruction error for the selected sample: {mean_reconstruction_error}")

# (d) Reconstruction error for all training samples

x_n = X_centered
z_n = x_n @ top_k_eigenvectors
x_tilde_n = z_n @ top_k_eigenvectors.T

input = x_n
target = x_tilde_n
squared_diff = (input - target) ** 2
mean_reconstruction_error = np.mean(squared_diff)

print(f"(d) Mean reconstruction error over all training samples: {mean_reconstruction_error}")

###### Solution 1-3: Reconstruction error with different number of components
print("\nSolution 1-3: Reconstruction error with different number of components\n")
print(f"(a) Randomly select 5 images and plot the reconstruction with "
      f" M = [2, 5, 10, 20, 50, 200] ")
M_list = np.array([2, 5, 10, 20, 50, 200])

# Data
X_train = X[:60000]
mean = np.mean(X_train, axis=0)  # Compute mean of each feature
X_centered = X_train - mean  # Center the data

J_M = []

fig, axes = plt.subplots(6, 10, figsize=(15, 10))

random_images_idx = np.sort(np.random.randint(60000, size=(1, 5)))
print(random_images_idx)

for j in range(M_list.shape[0]):
    M = M_list[j]  # Number of components to keep

    top_k_eigenvectors = PCA(X_centered, M)

    x_n = X_centered
    z_n = x_n @ top_k_eigenvectors
    x_tilde_n = z_n @ top_k_eigenvectors.T

    for i in range(X_train.shape[0]):

        if np.any(random_images_idx == i):
            (axes[j, int(np.where(random_images_idx == i)[1][0]) * 2].
             imshow((x_n[i] + mean).reshape(28, 28),cmap='gray'))
            (axes[j, int(np.where(random_images_idx == i)[1][0]) * 2].
             set_title("Original Image"))
            (axes[j, int(np.where(random_images_idx == i)[1][0]) * 2].
             axis('off'))

            # Plot reconstructed image
            (axes[j, int(np.where(random_images_idx == i)[1][0]) * 2 + 1].
             imshow((x_tilde_n[i] + mean).reshape(28, 28),cmap='gray'))
            (axes[j, int(np.where(random_images_idx == i)[1][0]) * 2 + 1].
             set_title("Reconstructed Image"))
            (axes[j, int(np.where(random_images_idx == i)[1][0]) * 2 + 1].
             axis('off'))

    input = x_n
    target = x_tilde_n
    squared_diff = (input - target) ** 2
    mean_reconstruction_error = np.mean(squared_diff)
    print(f"Mean reconstruction error for M = {M} is :"
          f" {mean_reconstruction_error}")
    J_M.append(mean_reconstruction_error)

plt.tight_layout()
plt.show()

print(f" (b) plotting mean construction error")
J_M_array = np.asarray(J_M)
plt.plot(np.arange(J_M_array.shape[0]), J_M_array)
plt.xlabel("M")
plt.ylabel("J_M")
plt.title("J_M at different M values [2, 5, 10, 20, 50, 200")
tick_positions = np.arange(J_M_array.shape[0])
# tick_labels = [f'{i}' for i in range(J_M_array.shape[0])]
tick_labels = ["2", "5", "10", "20", "50", "200"]
plt.xticks(tick_positions, tick_labels)
plt.show()

###### Solution 1-4: Reconstruction error of test dataset ######
print("\nSolution 1-4: Reconstruction error of test dataset\n")
X_train = X[:60000]
mean = np.mean(X_train, axis=0)  # Compute mean of each feature
X_centered = X_train - mean  # Center the data

y_test = y[60000:]
X_test = X[60000:]

X_test_centered = X_test - np.mean(X_test, axis=0)

J_M_test = []

for j in range(M_list.shape[0]):
    M = M_list[j]  # Number of components to keep
    top_k_eigenvectors = PCA(X_centered, M)
    x_n = X_test_centered
    z_n = x_n @ top_k_eigenvectors
    x_tilde_n = z_n @ top_k_eigenvectors.T

    reconstruction_errors = np.linalg.norm((x_n) - (x_tilde_n), axis=1)

    input = x_n
    target = x_tilde_n
    squared_diff = (input - target) ** 2
    mean_reconstruction_error = np.mean(squared_diff)
    print(f"Mean reconstruction error for M = {M} is :"
          f" {mean_reconstruction_error}")

    J_M_test.append(mean_reconstruction_error)

print(f" (a) plotting mean construction error of test dataset")
J_M_test_array = np.asarray(J_M_test)
plt.plot(np.arange(J_M_test_array.shape[0]), J_M_test_array)
plt.xlabel("M")
plt.ylabel("J_M_test")
plt.title("J_M_test at different M values [2, 5, 10, 20, 50, 200")
tick_positions = np.arange(J_M_array.shape[0])
tick_labels = ["2", "5", "10", "20", "50", "200"]
plt.xticks(tick_positions, tick_labels)
plt.show()

print("(b) Comparison between J_M of training dataset and test dataset")

plt.plot(np.arange(J_M_test_array.shape[0]), J_M_test_array, label="test")
plt.plot(np.arange(J_M_array.shape[0]), J_M_array, label="training")
plt.xlabel("M")
plt.ylabel("J_M")
plt.title("Comparison between J_M of training dataset and test dataset")
tick_positions = np.arange(J_M_array.shape[0])
tick_labels = ["2", "5", "10", "20", "50", "200"]
plt.xticks(tick_positions, tick_labels)
plt.legend()
plt.show()

###### Solution 1-5 2D Visualization ######

print("###### Solution 1-5 2D Visualization")
classes = [0, 1, 9]
samples_per_class = 100

indices = []
for digit in classes:
    digit_idxs = np.where(y == digit)[0]
    sampled_indices = np.random.choice(digit_idxs,
                                       samples_per_class,
                                       replace=False)
    indices.extend(sampled_indices)

X_selected = X[indices]
y_selected = y[indices]

mean = np.mean(X_selected, axis=0)  # Compute the mean of each feature
X_centered = X_selected - mean  # Center the data

# Covariance Matrix -> Eigen Vectors -> Get desired Eigen Vectors
top_2_eigenvectors  = PCA(X_centered, 2)

# Get z_n for desired eigen vectors
Z = X_centered @ top_2_eigenvectors
print("Data projected onto 2D space.")

# Plot
plt.figure(figsize=(10, 8))

colors = ['blue', 'brown', 'darkturquoise']
labels = ['Digit 0', 'Digit 1', 'Digit 9']

for i, digit in enumerate(classes):
    class_indices = np.where(y_selected == digit)[0]
    plt.scatter(Z[class_indices, 0], Z[class_indices, 1],
                label=labels[i], color=colors[i], alpha=0.6)

plt.title("2D Visualization of MNIST using PCA", fontsize=20)
plt.xlabel("Principal Component 1", fontsize=10)
plt.ylabel("Principal Component 2", fontsize=10)
plt.legend()
plt.grid(True)
plt.show()
