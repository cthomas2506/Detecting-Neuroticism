from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import codes.config as config
from codes.driver import sub_driver
from codes.utilities import bg_functions as bg
from codes.utilities import evaluation as evals


# Function to create majority tag model
def majority_tag_model(y_train):
    # Count occurrences of each class label
    label_counts = Counter(y_train)

    # Find the label with the highest count (majority tag)
    majority_tag = label_counts.most_common(1)[0][0]
    print(majority_tag)

    return majority_tag


# Function to create random assignment model
def random_assignment_model(unique_labels, num_samples):
    # Randomly select a label from unique labels for each sample
    random_labels = np.random.choice(unique_labels, size=num_samples)

    return random_labels


if __name__ == '__main__':
    # Global Parameters for fine tuning
    # --------------------------------------------------------------------------------------------------------------

    # to print out useful info while the code runs
    verbatim = True

    # use stratification
    stratify = True

    # preprocess data. refer to "utilities.bf_functions.preprocess_sentences"
    # to see the steps
    clean_data = True

    # some basic parameters
    test_size = 0.2

    # for uniformity
    seed = 19

    # load dataset and stop words
    dataset = config.PERSONALITY_DATASET
    stopwords_list = config.STOPWORDS

    # --------------------------------------------------------------------------------------------------------------

    # Driver functions
    # --------------------------------------------------------------------------------------------------------------

    # function to convert the CSV files to jsonl.gz files.
    # ----------------------------------
    bg.csv_to_jsonl_gz(config.PERSONALITY_CSV, config.PERSONALITY_DATASET)
    bg.csv_to_jsonl_gz(config.ESSAY_CSV, config.ESSAY_DATASET)

    # load data, clean it, split it into training and testing sets
    X_train, X_test, y_train, y_test, stop_words = sub_driver.fetch_data(
        dataset=dataset,
        stopwords=stopwords_list,
        clean_data=clean_data,
        verbatim=verbatim,
        test_size=test_size,
        seed=seed
    )

    # Create majority tag model
    majority_tag = majority_tag_model(y_train)

    # Assign majority tag to all test instances
    y_pred_majority = np.full_like(y_test, fill_value=majority_tag)

    # Evaluate majority tag model
    accuracy_majority = accuracy_score(y_test, y_pred_majority)

    print("Accuracy of majority tag model:", round(accuracy_majority, 3))
    print("Precision of majority tag model:", round(precision_score(y_test, y_pred_majority, pos_label='n'), 3))
    print("Recall of majority tag model:", round(recall_score(y_test, y_pred_majority, pos_label='n'), 3))
    print("F1 of majority tag model:", round(f1_score(y_test, y_pred_majority, pos_label='n'), 3))

    evals.plot_confusion_matrix(y_test, y_pred_majority)

    print("-"*50)



    # --------------------------------------------------------------------------------------------------------------

    # Get unique class labels
    unique_labels = np.unique(y_train)

    # Number of samples in the test set
    num_samples = len(y_test)

    # Create random assignment model
    y_pred_random = random_assignment_model(unique_labels, num_samples)

    accuracy_random = accuracy_score(y_test, y_pred_random)

    print("Accuracy of random assignment model:", round(accuracy_random, 3))
    print("Precision of random assignment model:", round(precision_score(y_test, y_pred_random, pos_label='n'), 3))
    print("Recall of random assignment model:", round(recall_score(y_test, y_pred_random, pos_label='n'), 3))
    print("F1 of random assignment model:", round(f1_score(y_test, y_pred_random, pos_label='n'), 3))

    evals.plot_confusion_matrix(y_test, y_pred_random)


