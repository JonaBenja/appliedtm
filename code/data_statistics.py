import pandas as pd
from string import punctuation
from collections import Counter

"""
TRAINING DATA
"""

baskerville = '../data/SEM-2012-SharedTask-CD-SCO-training-preprocessed.conll'

training_data = pd.read_csv(baskerville, encoding = 'utf-8', sep='\t')

tokens = training_data.iloc[:, 0]
labels = training_data.iloc[:, -1]
labeldict = Counter(labels)

n_tokens = [token for token in tokens if token not in punctuation]

n_cues = []
for label, token in zip(labels, tokens):
    if label != 'O':
        n_cues.append(token)

negation_cues = Counter(n_cues)

print('Training data:')
print('tokens', len(tokens), 'of which', len(tokens)-len(n_tokens), 'are punctuation')
print('Negation cues')
print(labeldict)
print(negation_cues.most_common(10))

"""
DEVELOPMENT DATA
"""

wistoria = '../data/SEM-2012-SharedTask-CD-SCO-dev-preprocessed.conll'
dev_data = pd.read_csv(wistoria, encoding = 'utf-8', sep='\t')

tokens = dev_data.iloc[:, 0]
labels = dev_data.iloc[:, -1]
labeldict = Counter(labels)

n_tokens = [token for token in tokens if token not in punctuation]

n_cues = []
for label, token in zip(labels, tokens):
    if label != 'O':
        n_cues.append(token)

negation_cues = Counter(n_cues)

print()
print('Development data:')
print('tokens', len(tokens), 'of which', len(tokens)-len(n_tokens), 'are punctuation')
print('Negation cues')
print(labeldict)
print(negation_cues.most_common(10))