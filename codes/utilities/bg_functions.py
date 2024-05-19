import codecs
import gzip
import json
import os
import re

import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split

import codes.config as config


def csv_to_jsonl_gz(file, destination):
    if not os.path.exists(destination):

        df = pd.read_csv(file, encoding="latin1")

        text_col = None
        df_selected = None

        if "personality" in file:
            text_col = "STATUS"
            df_selected = df[["#AUTHID", text_col, "cNEU"]]

        else:
            text_col = "TEXT"
            df_selected = df[["#AUTHID", text_col, "cNEU"]]

        # Write to JSONL file
        with gzip.open(destination, "wt") as jsonl_file:
            writer = jsonlines.Writer(jsonl_file)

            for index, row in df_selected.iterrows():
                data = {
                    "#AUTHID": row["#AUTHID"],
                    "STATUS": row[text_col],
                    "cNEU": row["cNEU"]
                }

                writer.write(data)

            print("file processed successfully")


def read_and_clean_lines(infile, verbatim):
    statuses = []
    labels = []

    with gzip.open(infile, 'rt') as f:
        # for line in tqdm(f):
        for line in f:
            data = json.loads(line)
            text = re.sub(r'\s+', ' ', data["STATUS"])

            statuses.append(text)
            labels.append(data["cNEU"])

    if verbatim:
        print("Read " + str(len(statuses)) + " documents and labels")
        print("-" * 30, end="\n")
        # print("Read {} labels".format(len(labels)))

    return statuses, labels


def split_training_set(lines, labels, test_size, random_seed):
    X_train, X_test, y_train, y_test = train_test_split(
        lines, labels,
        test_size=test_size, random_state=random_seed
    )

    return X_train, X_test, y_train, y_test


# TODO: experiment with different stop word modules
def load_stopwords(filename, words_list):
    if words_list == "standard":

        # NLTK
        # nltk.download('stopwords')
        # stopwords = list(nltk.corpus.stopwords.words('english'))

        # SpaCy
        # nlp = spacy.load("en_core_web_sm")
        # stopwords = nlp.Defaults.stop_words

        # SciKit-Learn
        stopwords = list(ENGLISH_STOP_WORDS)

        return stopwords

    elif words_list == "custom":
        stopwords = []
        with codecs.open(filename, 'r', encoding='ascii', errors='ignore') as fp:
            stopwords = fp.read().split('\n')
        return list(stopwords)


def remove_stop_words(stop_words, sentences):
    cleaned_sentences = []
    for sentence in sentences:
        words = sentence.split()
        cleaned_words = [word for word in words if word.lower() not in stop_words]
        cleaned_sentence = ' '.join(cleaned_words)
        cleaned_sentences.append(cleaned_sentence)
    return cleaned_sentences


def words_with_apostrophe(sentences):
    apostrophe_words = []
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            if "'" in word:
                apostrophe_words.append(word)
    return list(set(apostrophe_words))


def preprocess_sentences(sentences):
    preprocessed_sentences = []
    for sentence in sentences:
        # Lowercase every word
        sentence = sentence.lower()

        # Reduce multiple exclamation and question marks to just one
        sentence = re.sub(r'(\!{2,}|\?{2,}|\.{2,})', r'\1', sentence)

        # Automatically add a space after each punctuation
        sentence = re.sub(r'([.!?])', r'\1 ', sentence)

        # Remove all "*" characters
        sentence = sentence.replace('*', ' ')

        # Remove all double and single quotes
        sentence = sentence.replace('"', ' ')

        # Remove dashes '-'
        sentence = sentence.replace('-', ' ')

        sentence = sentence.replace(',', ' ')

        # Remove parentheses, brackets, and braces
        sentence = re.sub(r'[\(\)\[\]\{\}<>]', ' ', sentence)

        # Remove URLs
        sentence = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
                          sentence)
        preprocessed_sentences.append(sentence)
    return preprocessed_sentences


def expand_contractions(sentence):
    contractions_dict = {
        "here's": " here is ",
        "that's": " that is ",
        "it's": " it is ",
        "can't": " cannot ",
        "c'mon": " come on ",
        "don't": " do not ",
        "doesn't": " does not ",
        "won't": " will not ",
        "shouldn't": " should not ",
        "aren't": " are not ",
        "isn't": " is not ",
        "weren't": " were not ",
        "wouldn't": " would not ",
        "haven't": " have not ",
        "couldn't": " could not ",
        "hadn't": " had not ",
        "didn't": " did not ",
        "hasn't": " has not ",
        "wasn't": " was not ",
        "let's": " let us ",
        "i'll": " i will ",
        "she'll": " she will ",
        "he'll": " he will ",
        "we'll": " we will ",
        "they'll": " they will ",
        "i've": " i have ",
        "you've": " you have ",
        "we've": " we have ",
        "they've": " they have ",
        "i'd": " i would ",
        "you'd": " you would ",
        "he'd": " he would ",
        "she'd": " she would ",
        "it'd": " it would ",
        "we'd": " we would ",
        "they'd": " they would ",
        "i'm": " i am ",
        "you're": " you are ",
        "he's": " he is ",
        "she's": " she is ",
        "we're": " we are ",
        "they're": " they are ",
        "you'll": " you will ",
        "y'all": " you all ",
        "ya'll": " you all "
    }

    pattern = re.compile(r'\b(' + '|'.join(contractions_dict.keys()) + r')\b')
    expanded_sentence = pattern.sub(lambda x: contractions_dict[x.group()], sentence)

    return expanded_sentence


def plot_feature_importance(clf, top_n, X):
    print("hello world")

    # Get feature importances
    feature_importances = clf.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(feature_importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X.columns[i] for i in indices]

    # Slice the feature names and importances to select only the top n
    top_names = names[:top_n]
    top_importances = feature_importances[indices][:top_n]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.title("Top 5 Feature Importance - Variation")
    plt.bar(range(top_n), top_importances)
    plt.xticks(range(top_n), top_names, rotation=30)
    plt.show()


def fetch_ensenble_learning_dataset(dataset_type, seed, test_size):

    #  fetch entire dataset
    X, y = None, None

    text_col = None

    # Reading the dataset
    df = pd.read_csv(dataset_type, encoding="latin")

    # Dropping irrelevant columns
    if dataset_type == config.ESSAY_CSV:
        X = df.drop(columns=['cNEU', '#AUTHID'])
        text_col = "TEXT"

    elif dataset_type == config.PERSONALITY_CSV:
        X = df.drop(columns=['cNEU', 'DATE', '#AUTHID'])
        text_col = "STATUS"

    y = df['cNEU']

    # Performing one-hot encoding on binary flags
    X = pd.get_dummies(X, columns=['cEXT', 'cAGR', 'cCON', 'cOPN'], drop_first=True)

    # remove NaN rows
    X.dropna(inplace=True)
    y = y[X.index]

    X_train, X_test, y_train, y_test = split_training_set(X, y, test_size, seed)

    # then split the columns
    X_train1 = X_train.drop(columns=[text_col])
    X_test1 = X_test.drop(columns=[text_col])

    X_train2 = X_train[text_col]
    X_test2 = X_test[text_col]

    return X_train1, X_test1, X_train2, X_test2, y_train, y_test
