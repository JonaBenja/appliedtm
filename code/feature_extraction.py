import argparse
import pandas as pd

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
    output_file = input_file.replace('.tsv', '-features.tsv')

    # Read in preprocessed file
    input_data = pd.read_csv(input_file, encoding='utf-8', sep='\t')
    tokens = input_data.iloc[:, 0]
    labels = input_data.iloc[:, -1]

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for token, label in zip(tokens, labels):

            # Write token to
            outfile.write(token + '\t')

            # Write lemma feature value
            lemma = 'lemma'
            outfile.write(lemma + '\t')

            # Write POS feature value
            pos = 'POS'
            outfile.write(pos + '\t')

            # Write NE feature value
            ne = 'NE'
            outfile.write(ne + '\t')

            # Append label to line
            outfile.write(label + '\n')


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
