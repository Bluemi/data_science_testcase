import numpy as np
from utils import load_mnist, show_mnist_image


def main():
    images, labels = load_mnist()
    images = images.T

    u, s, v = np.linalg.svd(images[:, :1024])

    show_mnist_image(images[:, :64])

    s_limited = s.copy()
    s_limited[50:] = 0
    images_retrieved = u @ np.diag(s_limited) @ v[:784, :784]
    show_mnist_image(images_retrieved[:, :64])


if __name__ == '__main__':
    main()
