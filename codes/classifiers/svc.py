from sklearn.svm import SVC


def generate_classifier_and_params():
    classifier = SVC(kernel='linear')
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'gamma': ['scale', 'auto']
    }
    return classifier, param_grid
