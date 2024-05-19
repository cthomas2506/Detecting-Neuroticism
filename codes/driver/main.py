import codes.config as config
from codes.driver import sub_driver
from codes.utilities import bg_functions as bg

if __name__ == '__main__':
    # Global Parameters for fine tuning
    # ------------------------------------------------------------------------------------------------------------------

    # to print out useful info while the code runs
    verbatim = True

    # use stratification
    stratify = True

    # preprocess data. refer to "utilities.bf_functions.preprocess_sentences"
    # to see the steps
    clean_data = True

    # some basic parameters
    test_size = 0.2
    num_folds = 5

    # for uniformity
    seed = 19

    # list of ML classifications methods one can try
    # -----------------------------------------------
    # ml_model_to_try = ["Logistic Reg", "Random Forest", "SVM", "KNN"]
    ml_model_to_try = ["Logistic Reg"]

    # 3 different NLP methods to try
    # --------------------------------
    nlp_method = "Count Vectorizer"
    # nlp_method = "TF-IDF Vectorizer"
    # nlp_method = "Word Embeddings"

    # load dataset and stop words
    dataset = config.PERSONALITY_DATASET
    stopwords_list = config.STOPWORDS

    # type of evaluation - comment/uncomment to switch
    # --------------------
    mode = "Cross Validation"
    # mode = "Test Set Validation"

    # ------------------------------------------------------------------------------------------------------------------

    # Driver functions
    # ------------------------------------------------------------------------------------------------------------------

    # function to convert the CSV files to jsonl.gz files.
    # ----------------------------------
    bg.csv_to_jsonl_gz(config.PERSONALITY_CSV, config.PERSONALITY_DATASET)
    bg.csv_to_jsonl_gz(config.ESSAY_CSV, config.ESSAY_DATASET)

    # load data, clean it, split it into training and testing sets
    X_train, X_test, y_train, y_test, stop_words = sub_driver.fetch_data(
        dataset=dataset,
        stopwords=stopwords_list,
        clean_data=clean_data, verbatim=verbatim,
        test_size=test_size, seed=seed
    )

    # perform the needed training and testing
    sub_driver.train_and_test_models(
        X_train, X_test, y_train, y_test,
        stop_words=stop_words, mode=mode,
        nlp_method=nlp_method, ml_model_to_try=ml_model_to_try,
        verbatim=verbatim, stratify=stratify,
        num_folds=num_folds,
        seed=seed, test_size=test_size
    )