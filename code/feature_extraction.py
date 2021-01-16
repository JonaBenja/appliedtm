import argparse
import pandas as pd
from utils import *

"""
'../data/SEM-2012-SharedTask-CD-SCO-training-preprocessed.tsv'
'../data/SEM-2012-SharedTask-CD-SCO-dev-preprocessed.tsv'
"""


def write_features(input_file):
    """
    Function that adds features to the preprocessed data and writes it to an output file.
    :param input_file: path to the preprocessed data file
    :type input_file: string
    """
    # Prepare output file
    output_file = input_file.replace('.tsv', '-features.conll')

    # Read in preprocessed file
    input_data = pd.read_csv(input_file, encoding='utf-8', sep='\t')
    tokens = input_data.iloc[:, 0]
    labels = input_data.iloc[:, -1]

    # Defining header names
    feature_names = ["token",
                "lemma",
                "pos_tag",
                "punctuation",
                "gold_label"]

    lemmas = lemma_extraction(tokens)

    pos_tags = pos_extraction(tokens)

    prev_next_tokens = previous_and_next_token_extraction(tokens)
    prev_tokens, next_tokens = prev_next_tokens

    punctuation = is_punctuation(tokens)

    features_dict = {'token': tokens, 'lemma': lemmas, 'pos_tag': pos_tags, 'prev_token':prev_tokens,
                     'next_token': next_tokens, 'punctuation': punctuation, 'gold_label': labels}

    features_df = pd.DataFrame(features_dict, columns = feature_names)

    features_df.to_csv(output_file, sep='\t', index=False)

def main():
    # Set up command line parser
    parser = argparse.ArgumentParser(prog='feature_extraction.py',
                                     usage='python %(prog)s path_to_file',)
    parser.add_argument('input_file',
                        type=str,
                        help='file path to the input data to preprocess.'
                             'Example path: "../data/SEM-2012-SharedTask-CD-SCO-dev-simple.txt"')

    args = parser.parse_args()
    input_file = args.input_file

    # Write features to the feature file
    write_features(input_file)


if __name__ == '__main__':
    main()
