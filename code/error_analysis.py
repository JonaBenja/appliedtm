import pandas as pd
import argparse
from collections import defaultdict
def create_feature_dict(feature_list):
    feature_dict = dict()
    for feature in feature_list:
        feature_dict[feature] = defaultdict(int)

    return feature_dict

def error_analysis(inputfile):
    # Prepare variables and error dictionary
    features = ['token', 'lemma', 'pos_tag', 'prev_token', 'next_token', 'punctuation', 'affixes', 'n_grams']

    n_samples = 0
    n_errors = 0

    ft_error_dict = dict()
    tok_error_dict = defaultdict(list)

    # Go through all samples of development set
    for line in inputfile:
        components = line.rstrip('\n').split()
        if len(components) > 9:
            if n_samples > 0:

                # Find prediction of system and gold_label
                token = components[0]
                lemma = components[1]
                pos_tag = components[2]
                prev_token = components[3]
                next_token = components[4]
                punctuation = components[5]
                affixes = components[6]
                n_grams = list(components[7])
                label = components[-2]
                prediction = components[-1]

                # If they are not the same, the system made an error
                if prediction != label:
                    n_errors += 1
                    error_type = prediction + ' instead of ' + label

                    if error_type not in ft_error_dict:
                        feature_dict = create_feature_dict(features)
                        ft_error_dict[error_type] = feature_dict

                    for feature, value in zip(features, components):
                        ft_error_dict[error_type][feature][value] += 1

                    tok_error_dict[error_type].append([n_samples, prev_token, token, next_token])

            n_samples += 1

    for error_type in ft_error_dict:
        print('------------------------------------------------------------------------')
        print('-----> ', error_type)
        print(len(tok_error_dict[error_type]), 'error(s) of this type')
        print()
        for error in tok_error_dict[error_type]:
            print('Line number:', error[0])
            print('Preceding:', error[1])
            print('Current:', error[2])
            print('Next:', error[3])
            print()

        print('------------------------------------------------------------------------')
        for feature in ft_error_dict[error_type]:
            if feature in ['pos_tag', 'punctuation', 'affixes']:
                print(feature)
                print(ft_error_dict[error_type][feature])
                print()

def main():
    # Set up command line parser
    parser = argparse.ArgumentParser(prog='error_analysis.py',
                                     usage='python %(prog)s path_to_file',)
    parser.add_argument('input_file',
                        type=str,
                        help='file path to the input data to preprocess.'
                             'Example path: "../data/SEM-2012-SharedTask-CD-SCO-dev-simple-preprocessed-features-prediction1.conll"')

    args = parser.parse_args()
    input_file = args.input_file

    with open(input_file, 'r') as infile:
        error_analysis(infile)

if __name__ == '__main__':
    main()
