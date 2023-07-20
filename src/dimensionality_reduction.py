from utils import load_mnist, plot_embedding
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

NUM_IMAGES = 600


def main():
    images, labels = load_mnist()

    method = 'pca'
    if method == 'pca':
        embedding = PCA()
    elif method == 'mds':
        embedding = MDS(normalized_stress='auto')
    else:
        raise ValueError('Unknown method: {}'.format(method))
    transformed = embedding.fit_transform(images[:NUM_IMAGES])
    plot_embedding(transformed, labels[:NUM_IMAGES])


if __name__ == '__main__':
    main()
