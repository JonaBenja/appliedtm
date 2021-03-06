import argparse
import gensim
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer


def extract_word_embedding(token, word_embedding_model):
    """
    Function that returns the word embedding for a given token out of a
    distributional semantic model and a 300-dimension vector of 0s otherwise

    :param token: the token
    :param word_embedding_model: the distributional semantic model
    :type token: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors

    :returns a vector representation of the token
    """

    if token in word_embedding_model:
        vector = word_embedding_model[token]
    else:
        vector = [0]*300
    return vector


def combine_embeddings(data, word_embedding_model):
    """
    Extracts word embeddings for the token, previous token and next token and concatenates them

    :param data: a pandas dataframe
    :param word_embedding_model: the distributional semantic model
    :type data: pandas.core.frame.DataFrame
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors

    :returns a list containing are combined embeddings
    :rtype: list
    """
    embeddings = []
    for token, prev_token, next_token, lemma in zip(data['token'], data['prev_token'], data['next_token'], data['lemma']):

        # Extract embeddings for all token features
        token_vector = extract_word_embedding(token, word_embedding_model)
        prev_token_vector = extract_word_embedding(prev_token, word_embedding_model)
        next_token_vector = extract_word_embedding(next_token, word_embedding_model)
        lemma_vector = extract_word_embedding(lemma, word_embedding_model)

        # Concatenate the embeddings
        embeddings.append(np.concatenate((token_vector, prev_token_vector, next_token_vector, lemma_vector)))

    return embeddings


def make_sparse_features(data, feature_names):
    """
    Transforms traditional features into one-hot-encoded vectors

    :param data: a pandas dataframe
    :param feature_names: a list containing the header names of the traditional features
    :type data: pandas.core.frame.DataFrame
    :type feature_names: list

    :returns a vector representation of the traditional features
    :rtype: list
    """

    sparse_features = []
    for i in range(len(data)):

        # Prepare feature dictionary for each sample
        feature_dict = defaultdict(str)

        # Add feature values to dictionary
        for feature in feature_names:
            value = data[feature][i]
            feature_dict[feature] = value

        # Append all sample feature dictionaries
        sparse_features.append(feature_dict)

    return sparse_features


def combine_features(sparse, dense):
    """
    Combines sparse (one-hot-encoded) and dense (e.g. word embeddings) features
    into a combined feature set.

    :param sparse: one-hot representations of the traditional features
    :param dense: word embeddings of the token features
    :type sparse: list
    :type dense: list

    :returns a vector representation of all features combined
    :rtype: list

    """
    # Prepare containers
    combined_vectors = []
    sparse = np.array(sparse.toarray())

    # Concatanate vectors for each sample
    for index, vector in enumerate(sparse):
        combined_vector = np.concatenate((vector, dense[index]))
        combined_vectors.append(combined_vector)

    return combined_vectors


def train_classifier(x_train, y_train):
    """
    Trains the Multilayer Perceptron neural network

    :param x_train: training data
    :param y_train: training labels
    :type x_train: list
    :type y_train: list
    :returns a trained classifier
    :rtype: sklearn.neural_network._multilayer_perceptron.MLPClassifier
    """

    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=500, random_state=1)
    clf.fit(x_train, y_train)
    return clf


def load_data_embeddings(training_data_path, test_data_path, embedding_model_path):

    print('Loading word embedding model...')
    embedding_model = gensim.models.KeyedVectors.load_word2vec_format(
        embedding_model_path, binary=True)
    print('Done loading word embedding model')

    training = pd.read_csv(training_data_path, encoding='utf-8', sep='\t')
    training_labels = training['gold_label']

    test = pd.read_csv(test_data_path, encoding='utf-8', sep='\t')
    test_labels = test['gold_label']

    return training, training_labels, test, test_labels, embedding_model

def run_classifier(training, training_labels, test, word_embedding_model, sparse):
    """
    Function to create a classifier and train it on a training datafile.

    :param training_data_path: path to the training datafile
    :param test_data_path: path to the test datafile
    :param embedding_model_path: path to the embedding model

    :type training_data_path: string
    :type test_data_path: string
    :type embedding_model_path: string

    :returns a trained classifier, test data and test labels
    :rtype: sklearn.neural_network._multilayer_perceptron.MLPClassifier, list, list
    """

    # Extract embeddings for token, prev_token and next_token
    embeddings = combine_embeddings(training, word_embedding_model)

    # Extract and vectorize one-hot features
    sparse_features = make_sparse_features(training, sparse)
    vec = DictVectorizer()
    sparse_vectors = vec.fit_transform(sparse_features)

    # Combine both kind of features into training data
    training_data = combine_features(sparse_vectors, embeddings)

    # Train network
    print("Training classifier...")
    clf = train_classifier(training_data, training_labels)
    print("Done training classifier")

    # Extract embeddings for token, prev_token and next_token from test data
    embeddings = combine_embeddings(test, word_embedding_model)

    # Extract and vectorize one-hot features for test data
    sparse_features = make_sparse_features(test, sparse)
    sparse_vectors = vec.transform(sparse_features)

    test_data = combine_features(sparse_vectors, embeddings)

    return clf, test_data


def evaluation(test_labels, prediction):
    """
    Function to print f-score, precision and recall for each class of test data.
    Also prints confusion matrix

    :param test_labels: the test labels from the test set
    :param prediction: the prediction of the trained classifier on the test set
    :type test_labels: list
    :type prediction: list
    """
    metrics = classification_report(test_labels, prediction, digits=3)
    print(metrics)

    # Confusion matrix
    data = {'Gold': test_labels, 'Predicted': prediction}
    df = pd.DataFrame(data, columns=['Gold', 'Predicted'])

    confusion_matrix = pd.crosstab(df['Gold'], df['Predicted'], rownames=['Gold'], colnames=['Predicted'])
    print(confusion_matrix)
    print()


def main():
    # Set up command line parser
    parser = argparse.ArgumentParser(prog='mlp_morph.py',
                                     usage='python %(prog)s training_data_file test_data_file',
                                     description= 'This script trains two classifiers based on a Multilayer Perceptron. '
                                                  'Note that this script might run for around 1h.')

    arguments = ['training_data', 'test_data', 'embedding_model']
    helps = ['file path to the input data to preprocess. Example path: ../data/SEM-2012-SharedTask-CD-SCO-training-simple-features.conll',
             'file path to the input data to preprocess. Example path: ../data/SEM-2012-SharedTask-CD-SCO-dev-simple-features.conll',
             'file path to a pretrained embedding model. Example path: ../models/GoogleNews-vectors-negative300.bin']

    # Add arguments to command line parser
    for argument, help_message in zip(arguments, helps):
        parser.add_argument(argument, type=str, help=help_message)

    args = parser.parse_args()

    # Load arguments into variables
    training_data_path = args.training_data
    test_data_path = args.test_data
    embedding_model_path = args.embedding_model

    # Load data and the embedding model
    training, training_labels, test, test_labels, word_embedding_model = load_data_embeddings(training_data_path, test_data_path, embedding_model_path)

              # just traditional features (MLP)
    sparse = [["pos_tag", "punctuation"],
              # with morphological features (MLP-MORPH)
              ["pos_tag", "punctuation", "affixes", 'n_grams']]

    # Train classifiers
    for features in sparse:
        clf, test_data = run_classifier(training, training_labels, test, word_embedding_model, features)

        # Make prediction
        prediction = clf.predict(test_data)

        # Print evaluation
        print('-------------------------------------------------------')
        print("Evaluation of MLP system with the following sparse features:")
        print(features)
        evaluation(test_labels, prediction)


if __name__ == '__main__':
    main()
