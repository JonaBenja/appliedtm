import pandas as pd
import argparse
from string import punctuation
from collections import Counter

def print_statistics(input_file):
    """Function to compute statistics of a datafile"""
    data = pd.read_csv(input_file, encoding = 'utf-8', sep='\t')

    tokens = data.iloc[:, 0]
    labels = data.iloc[:, -1]
    labeldict = Counter(labels)

    n_tokens = [token for token in tokens if token not in punctuation]

    n_cues = []
    for label, token in zip(labels, tokens):
        if label != 'O':
            n_cues.append(token)

    negation_cues = Counter(n_cues)

    print('Data statistics:')
    print('tokens', len(tokens), 'of which', len(tokens)-len(n_tokens), 'are punctuation')
    print('Negation cues')
    print(labeldict)
    print(negation_cues.most_common(10))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file',
                        help='file path to the input data to compute statistics of. Example path: "../data/SEM-2012-SharedTask-CD-SCO-dev-simple-preprocessed.conll"')
    args = parser.parse_args()
    input_file = args.input_file

    print_statistics(input_file)

if __name__ == '__main__':
    main()
