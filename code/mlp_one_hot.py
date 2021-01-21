from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from collections import defaultdict
import pandas as pd

#test

"""
TRAINING
"""

feature_names = ["token",
                "lemma",
                "pos_tag",
                "prev_token",
                "next_token",
                "punctuation"]

baskerville = '../data/SEM-2012-SharedTask-CD-SCO-training-preprocessed-features.conll'
training = pd.read_csv(baskerville, encoding='utf-8', sep='\t')

training_data = []

for i in range(len(training)):
    feature_dict = defaultdict(str)
    for feature in feature_names:
        value = training[feature][i]
        feature_dict[feature] = value

    training_data.append(feature_dict)

print(len(training_data), training_data[:5])

training_labels = training['gold_label']

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

wistoria = '../data/SEM-2012-SharedTask-CD-SCO-dev-preprocessed-features.conll'
test = pd.read_csv(wistoria, encoding='utf-8', sep='\t')


test_data = []

for i in range(len(test)):
    feature_dict = defaultdict(str)
    for feature in feature_names:
        value = test[feature][i]
        feature_dict[feature] = value

    test_data.append(feature_dict)

print(len(test_data), test_data[:5])

test_labels = test['gold_label']

x_test = vec.transform(test_data)

prediction = clf.predict(x_test)

metrics = classification_report(test_labels, prediction, digits=3)
print(metrics)


errors = []
for y_pred, y_true in zip(prediction, test_labels):
    if y_pred != y_true:
        errors.append((y_pred, y_true))

print('Errors made (prediction, label):')
print(errors)

# One-hot encoded tokens
# 0.9952086097596934

# Word embeddings:
# 0.9954297508477075
