import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import codes.config as config
import codes.models.word_embeddings as we
from codes.classifiers import logistic_reg
from codes.utilities import bg_functions as bg
from codes.utilities import evaluation as evals

if __name__ == '__main__':

    seed = 19
    test_size = 0.2

    # -----------------------------------------------------
    # comment/uncomment to swap between embeddings

    glove_file = config.GLOVE_FILE_50D
    # glove_file = config.GLOVE_FILE_200D

    # -----------------------------------------------------------------------------------------------------------------
    # comment/uncomment to swap between datasets

    working_dataset = config.PERSONALITY_CSV
    # working_dataset = config.ESSAY_CSV
    # -----------------------------------------------------

    X_train1, X_test1, X_train2, X_test2, y_train, y_test = \
        bg.fetch_ensenble_learning_dataset(working_dataset, seed, test_size)

    stop_words = bg.load_stopwords(config.STOPWORDS, "custom")

    # -----------------------------------------------------------------------------------------------------------------
    model1 = RandomForestClassifier(random_state=seed)
    model1.fit(X_train1, y_train)

    words, word_to_vec_map = we.load_glove_model(glove_file)
    X_train_embeddings = we.glove_vectorizer(X_train2, word_to_vec_map)

    model2, param_grid = logistic_reg.generate_classifier_and_params()

    X_test_embeddings = we.glove_vectorizer(X_test2, word_to_vec_map)
    model2.fit(X_train_embeddings, y_train)
    # -----------------------------------------------------------------------------------------------------------------

    y_pred1 = model1.predict(X_test1)
    y_pred2 = model2.predict(X_test_embeddings)

    # -----------------------------------------------------------------------------------------------------------------

    num_samples = len(y_pred1)
    combined_pred = np.zeros_like(y_pred1)

    for i in range(num_samples):
        rand_num = np.random.rand()

        if rand_num <= 0.55:
            combined_pred[i] = y_pred1[i]
        else:
            combined_pred[i] = y_pred2[i]

    # -----------------------------------------------------------------------------------------------------------------

    accuracy = accuracy_score(y_test, combined_pred)
    precision = precision_score(y_test, combined_pred, pos_label='n')
    recall = recall_score(y_test, combined_pred, pos_label='n')
    f1 = f1_score(y_test, combined_pred, pos_label='n')

    print("Accuracy:", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1-score:", round(f1, 3))

    evals.plot_confusion_matrix(y_test, combined_pred)
