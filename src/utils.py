import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt


def greatest_divisor(n):
    start = int(np.ceil(np.sqrt(n)))
    for t in range(start, 1, -1):
        if n % t == 0:
            return t


def load_mnist():
    mnist_data = MNIST('data/mnist')
    images, labels = mnist_data.load_training()
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


def show_mnist_image(image):
    """
    Shows an image with shape [784,] or a stack of images with shape [784, N].
    :param image: Image or images as numpy ndarray
    """
    if len(image.shape) == 2:
        num_images = image.shape[1]
        side_length1 = greatest_divisor(num_images)
        while side_length1 is None or (side_length1 / (num_images // side_length1) > 2):
            num_images += 1
            side_length1 = greatest_divisor(num_images)
        side_length2 = num_images // side_length1
        images_reshaped = np.reshape(image, (28, 28, -1))

        # Create an empty big image with shape [56, 56]
        render_image = np.zeros((28 * side_length2, 28 * side_length1))

        # Iterate through each image in the array
        for i in range(image.shape[1]):
            # Calculate the starting row and column indices for each small image
            start_row = i // side_length1 * 28
            start_col = i % side_length1 * 28

            # Assign the small image to the corresponding region in the big image
            render_image[start_row:start_row + 28, start_col:start_col + 28] = images_reshaped[:, :, i]
    else:
        render_image = image.reshape((28, 28))
    plt.imshow(render_image, cmap='gray')
    plt.show()


def plot_embedding(embedding, labels=None, colors=None):
    """
    Plots the given embedding as scatter plot.
    :param embedding: An 2D-embedding with shape [N, 2]
    :param labels: Labels for the given embedding with shape [N,] or None
    :param colors: The colors to use for the plot
    """
    cmap = plt.cm.get_cmap('tab10')
    if labels is not None:
        c = labels
    elif colors is not None:
        c = colors
    else:
        c = None

    plt.scatter(embedding[:, 0], embedding[:, 1], c=c, cmap=cmap)

    if labels is not None:
        unique_labels = np.unique(labels)
        legend_handles = []
        legend_labels = []
        for label in unique_labels:
            legend_handles.append(plt.Line2D([], [], marker='o', color=cmap(label / 10), linestyle='None'))
            legend_labels.append(str(label))
        plt.legend(legend_handles, legend_labels)
    plt.show()
