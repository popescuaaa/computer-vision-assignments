"""

    Faces classification using principal components analysis for feature selection

"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA as RandomizedPCA  # For comparison purpose
from sklearn.svm import SVC

from argparse import ArgumentParser


def perform_pca(image: np.ndarray, n_components: int) -> np.ndarray:
    """
    Perform principal components analysis on the image

    :param image: image to perform pca on
    :param n_components: number of components to keep
    :return: image after pca
    """
    # Calculate mean
    mean = np.mean(image, axis=0)

    # Subtract mean from image
    image = image - mean

    # Calculate covariance matrix
    cov = np.cov(image.T)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Calculate projection matrix
    projection_matrix = eigenvectors[:, :n_components]

    # Project image onto the new basis
    image_projected = np.dot(image, projection_matrix)

    return image_projected


def plot_pca(image: np.ndarray, n_components: int) -> None:
    """
    Plot the image after pca

    :param image: image to plot
    :param n_components: number of components to keep
    :return: None
    """
    # Perform pca
    image_projected = perform_pca(image, n_components)

    # Plot image
    plt.imshow(image_projected, cmap='gray')
    plt.show()


def generate_datasets(train_dir: str, test_dir: str):
    # Create an image dataset
    train_image_dataset, test_image_dataset = [], []
    train_features, test_features = [], []
    train_labels, test_labels = [], []

    for train_filename in os.listdir(train_dir):
        image_class = int(train_filename.split('_')[0])
        image_illumination = int(train_filename.split('_')[1].split('.')[0])
        image_path = os.path.join(train_dir, train_filename)
        image = plt.imread(image_path)

        train_features.append(image.flatten())
        train_labels.append(image_class)
        train_image_dataset.append((image, image_class, image_illumination))

    # Convert to numpy array
    train_features = np.array(train_features)
    train_labels = np.array(train_labels)

    for test_filename in os.listdir(test_dir):
        image_class = int(test_filename.split('_')[0])
        image_illumination = int(test_filename.split('_')[1].split('.')[0])
        image_path = os.path.join(test_dir, test_filename)
        image = plt.imread(image_path)

        test_features.append(image.flatten())
        test_labels.append(image_class)
        test_image_dataset.append((image, image_class, image_illumination))

    # Convert to numpy array
    test_features = np.array(train_features)
    test_labels = np.array(train_labels)

    return {
        "train": {
            "features": train_features,
            "labels": train_labels,
            "image_dataset": train_image_dataset
        },
        "test": {
            "features": test_features,
            "labels": test_labels,
            "image_dataset": test_image_dataset
        }
    }


if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument('-num_components', '--num_components', type=int, required=False,
                        help='Number of components to keep', default=50)

    parser.add_argument('-classifier', '--classifier', type=str, required=False,
                        help='Classifier to use', default='svm')

    args = parser.parse_args()
    num_components = args.num_components
    classifier = args.classifier

    train_dir = 'Train'
    test_dir = 'Test'

    # Generate datasets (already split into train and test)
    datasets = generate_datasets(train_dir=train_dir, test_dir=test_dir)

    train = datasets['train']
    test = datasets['test']

    # Perform pca
    train_features_pca = RandomizedPCA(n_components=num_components).fit_transform(train['features'])
    test_features_pca = RandomizedPCA(n_components=num_components).fit_transform(test['features'])

    """
        Now we will reshape our PCA components and define eigenfaces, 
        which is the name given to a set of eigenvectors 
        when used in the computer vision problem of human face recognition:
    
    """
    # Reshape the eigenvectors
    eigenfaces = train_features_pca.reshape(num_components, -1)
    print('Eigenvalues shape:', eigenfaces.shape)

    # Train a classifier
    print("Fitting the classifier to the training set")
    param_grid = {
        'C': [1e3, 5e3, 1e4, 5e4, 1e5],
        'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
    }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(train_features_pca, train['labels'])
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    # Predict
    pred_labels = clf.predict(test_features_pca)
    print(classification_report(test['labels'], pred_labels, target_names=[str(i) for i in range(1, 6)]))
