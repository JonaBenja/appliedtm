from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict
import numpy as np
import pandas as pd
import gensim


print('Loading word embedding model...')
word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format('../models/GoogleNews-vectors-negative300.bin', binary=True)
print('Done loading word embedding model')

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

"""
TRAINING
"""

baskerville = '../data/SEM-2012-SharedTask-CD-SCO-training-preprocessed-features.conll'
training = pd.read_csv(baskerville, encoding='utf-8', sep='\t')

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


sparse = ["pos_tag",
          "punctuation"]

embeddings = combine_embeddings(training, word_embedding_model)
sparse_features = make_sparse_features(training, sparse)

vec = DictVectorizer()
sparse_vectors = vec.fit_transform(sparse_features)

training_data = combine_features(sparse_vectors, embeddings)

training_labels = [label for label in training['gold_label']]


x_train = training_data
y_train = training_labels

clf = MLPClassifier(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=(300), random_state=1)
print("Training network...")
clf.fit(x_train, y_train)
print("Done training network")

#TESTING

wistoria = '../data/SEM-2012-SharedTask-CD-SCO-dev-preprocessed-features.conll'
test = pd.read_csv(wistoria, encoding='utf-8', sep='\t')

embeddings = combine_embeddings(test, word_embedding_model)
sparse_features = make_sparse_features(test, sparse)

sparse_vectors = vec.transform(sparse_features)

test_data = combine_features(sparse_vectors, embeddings)


test_labels = test['gold_label']

prediction = clf.predict(test_data)

metrics = classification_report(test_labels, prediction, digits=3)
print(metrics)

"""
errors = []
i = 0
for y_pred, y_true, token in zip(prediction, test_labels, test.iloc[:, 0]):
    i += 1
    if y_pred != 'O' and y_true == 'O':
        errors.append({token: [y_pred, y_true, i]})
    elif y_pred == 'O' and y_true != 'O':
        print(token, y_pred, y_true, i)

print(errors)
"""
# One-hot encoded tokens
# 0.9952086097596934

# Word embeddings:
# 0.9954297508477075
