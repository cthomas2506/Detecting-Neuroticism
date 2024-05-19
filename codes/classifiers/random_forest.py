from sklearn.ensemble import RandomForestClassifier


def generate_classifier_and_params(seed):
    classifier = RandomForestClassifier(random_state=seed)
    param_grid = {
        'n_estimators': [10, 20, 30, 50, 100],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    return classifier, param_grid
