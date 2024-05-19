from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, KFold

from codes.classifiers import knn, logistic_reg, random_forest, svc
from codes.utilities import evaluation


def convert_text_into_features(X, stopwords_arg, analyzefn="word", range=(1, 2)):
    training_vectorizer = CountVectorizer(stop_words=stopwords_arg,
                                          analyzer=analyzefn,
                                          lowercase=True,
                                          ngram_range=range)
    X_features = training_vectorizer.fit_transform(X)

    return X_features, training_vectorizer


def count_vectorizer_analysis(
        X_train, X_test, y_train, y_test,
        stop_words, test_size,
        mode, ml_model_to_try,
        verbatim, stratify,
        num_folds, seed):

    classifier = None
    param_grid = None

    X_features_train, training_vectorizer = convert_text_into_features(
        X_train, stop_words, "word",
        range=(1, 2)
    )

    if verbatim:
        print("{}-fold cross-validation, stratification={}".format(num_folds, stratify))

    if stratify:
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    else:
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    for ml_model in ml_model_to_try:
        print("Working with:", ml_model)

        if ml_model == "Logistic Reg":
            classifier, param_grid = logistic_reg.generate_classifier_and_params()

        elif ml_model == "SVM":
            classifier, param_grid = svc.generate_classifier_and_params()

        elif ml_model == "Random Forest":
            classifier, param_grid = random_forest.generate_classifier_and_params(seed)

        elif ml_model == "KNN":
            classifier, param_grid = knn.generate_classifier_and_params()

        if mode == "Cross Validation":
            evaluation.k_fold_cross_val_evaluation(classifier=classifier, param_grid=param_grid,
                                                   kfold=kfold, verbatim=verbatim,
                                                   X_train=X_features_train, y_train=y_train)

        elif mode == "Test Set Validation":
            print("Evaluating on the test set")

            evaluation.test_set_evaluation(training_vectorizer=training_vectorizer, classifier=classifier,
                                           X_train=X_features_train, y_train=y_train,
                                           X_test=X_test, y_test=y_test)
