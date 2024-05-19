import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

import codes.config as config
from codes.driver import sub_driver


def load_embeddings_and_build_model(tokenizer, glove_file_path, embedding_dim):

    # Load GloVe embeddings into a dictionary
    embeddings_index = {}
    with open(glove_file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # Create an embedding matrix
    word_index = tokenizer.word_index

    num_words_to_include = min(len(word_index) + 1, 30000)

    embedding_matrix = np.zeros((num_words_to_include, embedding_dim))

    for word, index in word_index.items():
        if index < num_words_to_include:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

    return embedding_matrix, num_words_to_include


def build_cnn_model(num_words, embedding_matrix, max_sequence_length, embedding_dim):
    model = Sequential()

    # Add embedding layer
    model.add(
        Embedding(input_dim=num_words, output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  input_length=max_sequence_length,
                  trainable=False)
    )

    # Add convolutional layers
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # Flatten layer
    model.add(Flatten())

    # Add fully connected layers
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(2, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(model.summary())

    return model


def tokenize_and_prepare_dataset(X_train, X_test, y_train, y_test, tokenizer):
    unique_labels = set(y_train)
    label_to_index = {label: float(i) for i, label in enumerate(unique_labels)}

    y_train = np.array([label_to_index[label] for label in y_train])
    y_test = np.array([label_to_index[label] for label in y_test])

    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    max_sequence_length = max([len(sequence) for sequence in X_train])

    X_train = pad_sequences(X_train, maxlen=max_sequence_length)
    X_test = pad_sequences(X_test, maxlen=max_sequence_length)

    return X_train, X_test, y_train, y_test, max_sequence_length


if __name__ == '__main__':
    # ---------------------------------------
    # please use either or.

    glove_file_path = config.GLOVE_FILE_200D
    embedding_dim = 200

    # glove_file_path = config.GLOVE_FILE_50D
    # embedding_dim = 50

    # ---------------------------------------
    # Comment/uncomment to swap datasets

    dataset = config.PERSONALITY_DATASET
    # dataset = config.ESSAY_DATASET

    # ---------------------------------------

    X_train, X_test, y_train, y_test, _ = sub_driver.fetch_data(
        dataset=dataset,
        stopwords=config.STOPWORDS,
        clean_data=True, verbatim=False,
        test_size=0.2, seed=config.SEED
    )

    tokenizer = Tokenizer()
    X_train, X_test, y_train, y_test, max_sequence_length = tokenize_and_prepare_dataset(
        X_train, X_test,
        y_train, y_test,
        tokenizer
    )

    embedding_matrix, num_words = load_embeddings_and_build_model(
        tokenizer, glove_file_path, embedding_dim
    )

    model = build_cnn_model(num_words, embedding_matrix, max_sequence_length, embedding_dim)

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=30, batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping_callback]
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f'Test accuracy: {accuracy:.4f}')
