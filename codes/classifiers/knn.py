from sklearn.neighbors import KNeighborsClassifier


def generate_classifier_and_params():
    classifier = KNeighborsClassifier()
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    return classifier, param_grid
