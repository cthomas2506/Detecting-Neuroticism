import codes.config as config
from codes.models.count_vectorizer import count_vectorizer_analysis
from codes.models.tf_idf import tfidf_vectorizer_analysis
from codes.models.word_embeddings import glove_vectorizer_analysis
from codes.utilities import bg_functions as bg


# load data, clean it, split it into training and testing sets
def fetch_data(dataset, stopwords, clean_data, verbatim, test_size, seed):

    # loading stop words
    stop_words = bg.load_stopwords(
        stopwords,
        words_list="standard"
    )

    # read the lines from the jsonl.gz file
    X, y = bg.read_and_clean_lines(
        dataset,
        verbatim=verbatim
    )

    # option to perform an additional cleaning step as a part of preprocessing
    if clean_data:
        X = bg.preprocess_sentences(X)
        X = [bg.expand_contractions(sentence) for sentence in X]

    # split into training and testing sets
    X_train, X_test, y_train, y_test = bg.split_training_set(
        X, y,
        test_size=test_size,
        random_seed=seed
    )

    return X_train, X_test, y_train, y_test, stop_words


# perform:
#       1. k-fold cross validation on training set
#       2. test set validation

# using:
#       1. Count Vectorizer method
#       2. TF-IDF vectorizer method
#       3. Word Embeddings method

# with the classifier options:
#       1. Logistic regression classifier
#       2. Random Forest Classifier
#       3. KNN classifier
#       4. Support Vector Classifier

def train_and_test_models(
        X_train, X_test, y_train, y_test,
        stop_words, test_size,
        mode, nlp_method, ml_model_to_try,
        verbatim, stratify,
        num_folds, seed):

    if nlp_method == "Count Vectorizer":
        count_vectorizer_analysis(
            X_train, X_test, y_train, y_test,
            stop_words, test_size,
            mode, ml_model_to_try,
            verbatim, stratify,
            num_folds, seed
        )

    elif nlp_method == "TF-IDF Vectorizer":
        tfidf_vectorizer_analysis(
            X_train, X_test, y_train, y_test,
            stop_words, test_size,
            mode, ml_model_to_try,
            verbatim, stratify,
            num_folds, seed)

    elif nlp_method == "Word Embeddings":
        glove_file = config.GLOVE_FILE_50D
        glove_vectorizer_analysis(X_train, X_test, y_train, y_test,
                                  mode=mode,
                                  ml_model_to_try=ml_model_to_try,
                                  verbatim=True, stratify=True,
                                  num_folds=num_folds, seed=seed,
                                  glove_file=glove_file, test_size=test_size)
    else:
        return