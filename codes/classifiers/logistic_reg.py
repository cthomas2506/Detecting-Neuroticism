from sklearn.linear_model import LogisticRegression


def generate_classifier_and_params():
    classifier = LogisticRegression(solver='liblinear')
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2']
    }
    return classifier, param_grid
