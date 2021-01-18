from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
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

baskerville = '../data/SEM-2012-SharedTask-CD-SCO-training-preprocessed.tsv'
training = pd.read_csv(baskerville, encoding='utf-8', sep='\t')

training_data = []
for token in training.iloc[:, 0]:
    word_embedding = extract_word_embedding(token, word_embedding_model)
    training_data.append(word_embedding)

training_labels = [label for label in training.iloc[:, -1]]

print(len(training_data), len(training_labels))

x_train = training_data
y_train = training_labels

clf = MLPClassifier(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=(300), random_state=1)

print("Training network...")
clf.fit(x_train, y_train)
print("Done training network")

"""
#TESTING
"""

wistoria = '../data/SEM-2012-SharedTask-CD-SCO-dev-preprocessed.tsv'
test = pd.read_csv(wistoria, encoding='utf-8', sep='\t')

test_data = []
for token in test.iloc[:, 0]:
    word_embedding = extract_word_embedding(token, word_embedding_model)
    test_data.append(word_embedding)

test_labels = list(test.iloc[:, -1])

print(len(test_data), len(test_labels))

prediction = clf.predict(test_data)

metrics = classification_report(test_labels, prediction)
print(metrics)

errors = []
i = 0
for y_pred, y_true, token in zip(prediction, test_labels, test.iloc[:, 0]):
    i += 1
    if y_pred != 'O' and y_true == 'O':
        errors.append({token: [y_pred, y_true, i]})
    elif y_pred == 'O' and y_true != 'O':
        print(token, y_pred, y_true, i)

print(errors)

# One-hot encoded tokens
# 0.9952086097596934

# Word embeddings:
# 0.9954297508477075