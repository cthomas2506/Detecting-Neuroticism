import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

from codes.classifiers import knn, logistic_reg, random_forest, svc
from codes.utilities import evaluation


def load_glove_model(glove_file):
    print("Loading GloVe model...")

    with open(glove_file, 'r', encoding='utf-8') as f:
        words = set()
        word_to_vec_map = {}

        for line in f:
            line = line.strip().split()

            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

    print("Done.")

    return words, word_to_vec_map


def document_embedding(doc, word_to_vec_map):
    words = doc.split()
    embedding = np.zeros(word_to_vec_map["a"].shape)

    for w in words:
        embedding += word_to_vec_map.get(w.lower(), np.zeros(word_to_vec_map["a"].shape))

    return embedding / len(words)


def glove_vectorizer(X, word_to_vec_map):
    X_embeddings = [document_embedding(doc, word_to_vec_map) for doc in X]

    return np.array(X_embeddings)


def glove_vectorizer_analysis(X_train, X_test, y_train, y_test,
                              mode,
                              ml_model_to_try,
                              verbatim, stratify,
                              num_folds, seed,
                              glove_file,
                              test_size):
    words, word_to_vec_map = load_glove_model(glove_file)

    classifier = None

    # print(X_train)
    X_train_embeddings = glove_vectorizer(X_train, word_to_vec_map)

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

        if verbatim:
            print("Training Split: {}".format(round(1 - test_size, 2)))

        if mode == "Cross Validation":
            evaluation.k_fold_cross_val_evaluation(classifier=classifier, param_grid=param_grid,
                                                   kfold=kfold, verbatim=verbatim,
                                                   X_train=X_train_embeddings, y_train=y_train)

        elif mode == "Test Set Validation":
            print("Evaluating on the test set")
            X_test_embeddings = glove_vectorizer(X_test, word_to_vec_map)

            evaluation.test_set_evaluation(training_vectorizer=None, classifier=classifier,
                                           X_train=X_train_embeddings, y_train=y_train,
                                           X_test=X_test_embeddings, y_test=y_test)