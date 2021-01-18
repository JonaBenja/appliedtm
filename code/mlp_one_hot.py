from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
import pandas as pd
import gensim

"""
TRAINING
"""

baskerville = '../data/SEM-2012-SharedTask-CD-SCO-training-preprocessed.tsv'
training = pd.read_csv(baskerville, encoding='utf-8', sep='\t')

training_data = [{'token': token} for token in training.iloc[:, 0]]
training_labels = [label for label in training.iloc[:, -1]]

print(len(training_data), len(training_labels))

vec = DictVectorizer()
x_train = vec.fit_transform(training_data)

y_train = training_labels

clf = MLPClassifier(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=(15), random_state=1)

print("Training network...")
clf.fit(x_train, y_train)
print("Done training network")

"""
#TESTING
"""

wistoria = '../data/SEM-2012-SharedTask-CD-SCO-dev-preprocessed.tsv'
test = pd.read_csv(wistoria, encoding='utf-8', sep='\t')

test_data = [{'token': token} for token in test.iloc[:, 0]]
test_labels = list(test.iloc[:, -1])

x_test = vec.transform(test_data)

prediction = clf.predict(x_test)

accuracy = accuracy_score(test_labels, prediction, normalize=False)

print(accuracy, "/", len(test_labels))
print(accuracy/len(test_labels))

errors = []
for y_pred, y_true in zip(prediction, test_labels):
    if y_pred != y_true:
        errors.append((y_pred, y_true))

print(errors)

# One-hot encoded tokens
# 0.9952086097596934

# Word embeddings:
# 0.9954297508477075
