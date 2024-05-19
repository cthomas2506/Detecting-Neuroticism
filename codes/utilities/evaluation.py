import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score


def k_fold_cross_val_evaluation(classifier, param_grid, kfold, X_train, y_train, verbatim):
    if verbatim:
        print("Evaluating using cross-validation and hyperparameter tuning")

    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=kfold, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print("Best hyperparameters:", best_params)

    classifier = grid_search.best_estimator_

    accuracy_scores = cross_val_score(
        classifier,
        X_train, y_train,
        cv=kfold, scoring='accuracy',
    )

    precision_scores = cross_val_score(
        classifier,
        X_train, y_train,
        cv=kfold, scoring='precision_macro'
    )

    recall_scores = cross_val_score(
        classifier,
        X_train, y_train,
        cv=kfold, scoring='recall_macro'
    )

    f1_scores = cross_val_score(
        classifier,
        X_train, y_train,
        cv=kfold, scoring='f1_macro'
    )

    acc_means = round(np.mean(accuracy_scores), 5)
    acc_std_devs = round(np.std(accuracy_scores), 5)
    prec_means = round(np.mean(precision_scores), 5)
    recall_means = round(np.mean(recall_scores), 5)
    f1_means = round(np.mean(f1_scores), 5)

    if verbatim:
        print("\n\tmean accuracy = {}, "
              "\n\tmean precision = {}, "
              "\n\tmean recall = {}, "
              "\n\tmean F1-score = {}, "
              "\n\tstdev accuracy = {}\n\n"
              .format(acc_means, prec_means, recall_means, f1_means, acc_std_devs))


def plot_confusion_matrix(y_test, y_pred):

    cm = confusion_matrix(y_test, y_pred)

    # print(cm)

    plt.figure(figsize=(4, 3))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_test)))
    plt.xticks(tick_marks, np.unique(y_test))
    plt.yticks(tick_marks, np.unique(y_test))

    fmt = 'd'
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="black")

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()


def test_set_evaluation(training_vectorizer, classifier, X_train, y_train, X_test, y_test):
    # Word Embeddings
    if training_vectorizer is None:
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

    # TF-IDF and Count Vectorizer
    else:
        X_features_test = training_vectorizer.transform(X_test)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_features_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='n')
    recall = recall_score(y_test, y_pred, pos_label='n')
    f1 = f1_score(y_test, y_pred, pos_label='n')

    print("Accuracy:", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1-score:", round(f1, 3))

    plot_confusion_matrix(y_test, y_pred)


