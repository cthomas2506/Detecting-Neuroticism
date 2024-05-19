# Required Libraries
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import seaborn as sns
import spacy
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#TODO: Library requires work!

# Document Statistics
def document_statistics(documents):
    num_documents = len(documents)
    avg_doc_length = np.mean([len(doc.split()) for doc in documents])
    max_doc_length = np.max([len(doc.split()) for doc in documents])
    min_doc_length = np.min([len(doc.split()) for doc in documents])
    print("Number of documents:", num_documents)
    print("Average document length:", avg_doc_length)
    print("Maximum document length:", max_doc_length)
    print("Minimum document length:", min_doc_length)

    return max_doc_length


# Class Distribution
def class_distribution(labels):
    label_counts = Counter(labels)
    plt.bar(label_counts.keys(), label_counts.values())
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title('Class Distribution')
    plt.show()


# Word Frequency Analysis
def word_frequency_analysis(documents):
    tokenizer = RegexpTokenizer(r'\w+')
    all_words = [word.lower() for doc in documents for word in tokenizer.tokenize(doc)]
    fdist = FreqDist(all_words)
    plt.figure(figsize=(10, 5))
    fdist.plot(30, cumulative=False)
    plt.show()


# Stopwords and Punctuation
def remove_stopwords_and_punctuation(documents):
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    clean_documents = []
    for doc in documents:
        words = tokenizer.tokenize(doc)
        clean_words = [word for word in words if word.lower() not in stop_words]
        clean_doc = ' '.join(clean_words)
        clean_documents.append(clean_doc)
    return clean_documents


# TFIDF
def tfidf_representation(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix, vectorizer.get_feature_names_out()


# Word Embedding Visualization (using PCA)
def word_embedding_visualization(documents):
    word_list = [word_tokenize(doc.lower()) for doc in documents]
    model = Word2Vec(word_list, min_count=1)
    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    plt.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    plt.show()


# Document Similarity
def document_similarity(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    sns.heatmap(cosine_sim, annot=True, fmt=".2f", cmap='coolwarm')
    plt.show()


# Named Entity Recognition
def named_entity_recognition(documents):
    nlp = spacy.load("en_core_web_sm")
    for doc in documents:
        doc_entities = nlp(doc)
        for entity in doc_entities.ents:
            print(entity.text, entity.label_)


# Topic Modeling
def topic_modeling(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    svd = TruncatedSVD(n_components=2)
    svd_matrix = svd.fit_transform(tfidf_matrix)
    print(svd_matrix)

if __name__ == '__main__':

    # Download NLTK resources
    nltk.download('punkt')
    nltk.download('stopwords')

    # Sample text data
    documents = ["This is the first document.",
                 "This document is the second document.",
                 "And this is the third one.",
                 "Is this the first document?"]


    # document_statistics(documents)


    # labels = ['n', 'y', 'n', 'y']  # Sample labels for two classes
    # class_distribution(labels)
    #
    # word_frequency_analysis(documents)
    #
    # clean_documents = remove_stopwords_and_punctuation(documents)
    # print("Documents after removing stopwords and punctuation:", clean_documents)
    #
    # tfidf_matrix, feature_names = tfidf_representation(clean_documents)
    # print("TFIDF Matrix:")
    # print(tfidf_matrix.toarray())
    # print("Feature Names:", feature_names)
    #
    # word_embedding_visualization(documents)
    #
    # document_similarity(documents)
    #
    # named_entity_recognition(documents)
    #
    # topic_modeling(documents)



