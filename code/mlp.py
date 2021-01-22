from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict
import numpy as np
import pandas as pd
import gensim
import argparse


def extract_word_embedding(token, word_embedding_model):
    '''
    Function that returns the word embedding for a given token out of a distributional semantic model and a 300-dimension vector of 0s otherwise

    :param token: the token
    :param word_embedding_model: the distributional semantic model
    :type token: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors

    :returns a vector representation of the token
    '''
    if token in word_embedding_model:
        vector = word_embedding_model[token]
    else:
        vector = [0]*300
    return vector

def combine_embeddings(data, word_embedding_model):
    embeddings = []
    for token, prev_token, next_token, lemma in zip(data['token'], data['prev_token'], data['next_token'], data['lemma']):

        token_vector = extract_word_embedding(token, word_embedding_model)
        prev_token_vector = extract_word_embedding(prev_token, word_embedding_model)
        next_token_vector = extract_word_embedding(next_token, word_embedding_model)
        lemma_vector = extract_word_embedding(lemma, word_embedding_model)

        embeddings.append(np.concatenate((token_vector, prev_token_vector, next_token_vector, lemma_vector)))

    return embeddings


def make_sparse_features(training_data, feature_names, test=False):
    sparse_features = []
    for i in range(len(training_data)):
        feature_dict = defaultdict(str)
        for feature in feature_names:
            value = training_data[feature][i]
            feature_dict[feature] = value

        sparse_features.append(feature_dict)

    return sparse_features


def combine_features(sparse, dense):
    combined_vectors = []
    for index, vector in enumerate(dense):
            combined_vector = np.concatenate((vector, dense[index]))
            combined_vectors.append(combined_vector)

    return combined_vectors

def train_classifier(X_train, y_train):
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(300), random_state=1)
    clf.fit(X_train, y_train)
    return clf

def main():
    # Set up command line parser
    parser = argparse.ArgumentParser(prog='mlp_word_embeddings.py',
                                     usage='python %(prog)s training_data_file test_data_file',)
    parser.add_argument('training_data',
                        type=str,
                        help='file path to the input data to preprocess.'
                             'Example path: "../data/SEM-2012-SharedTask-CD-SCO-training-preprocessed.conll"')

    parser.add_argument('test_data',
                        type=str,
                        help='file path to the input data to preprocess.'
                             'Example path: "../data/SEM-2012-SharedTask-CD-SCO-dev-preprocessed.conll"')

    args = parser.parse_args()

    print('Loading word embedding model...')
    word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format(
        '../models/GoogleNews-vectors-negative300.bin', binary=True)
    print('Done loading word embedding model')

    sparse = ["pos_tag",
              "punctuation"]

    # Load training data
    training_data = args.training_data
    training = pd.read_csv(training_data, encoding='utf-8', sep='\t')

    # Extract embeddings for token, prev_token and next_token
    embeddings = combine_embeddings(training, word_embedding_model)

    # Extract and vectorize one-hot features
    sparse_features = make_sparse_features(training, sparse)
    vec = DictVectorizer()
    sparse_vectors = vec.fit_transform(sparse_features)

    # Combine both kind of features into training data
    training_data = combine_features(sparse_vectors, embeddings)
    training_labels = [label for label in training['gold_label']]

    # Train network
    print("Training classifier...")
    clf = train_classifier(training_data, training_labels)
    print("Done training classifier")

    # Load test data
    test_data = args.test_data
    test = pd.read_csv(test_data, encoding='utf-8', sep='\t')

    # Extract embeddings for token, prev_token and next_token from test data
    embeddings = combine_embeddings(test, word_embedding_model)

    # Extract and vectorize one-hot features
    sparse_features = make_sparse_features(test, sparse)
    sparse_vectors = vec.transform(sparse_features)

    # Combine both kind of features into training data
    test_data = combine_features(sparse_vectors, embeddings)
    test_labels = test['gold_label']

    # Make prediction
    prediction = clf.predict(test_data)

    # Evaluate
    metrics = classification_report(test_labels, prediction, digits=3)
    print(metrics)


if __name__ == '__main__':
    main()


# '../data/SEM-2012-SharedTask-CD-SCO-training-preprocessed-features.conll'
# '../data/SEM-2012-SharedTask-CD-SCO-dev-preprocessed-features.conll'
#'../data/SEM-2012-SharedTask-CD-SCO-test-cardboard-preprocessed-features.conll'