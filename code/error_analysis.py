import pandas as pd
import argparse
from collections import defaultdict

def error_analysis(inputfile):
    # Prepare variables and error dictionary
    features = ['token', 'lemma', 'pos_tag', 'prev_token', 'next_token', 'punctuation', 'affixes', 'n_grams']

    n_samples = 0
    n_errors = 0
    fp_dict = dict()
    fn_dict = dict()


    for feature in features:
        fp_dict[feature] = defaultdict(int)
        fn_dict[feature] = defaultdict(int)

    # Go through all samples of development set
    for line in inputfile:
        n_samples += 1
        components = line.rstrip('\n').split()

        # Find prediction of system and gol_label
        prediction = components[-1]
        label = components[-2]

        # If they are not the same, the system made an error
        if prediction != 'O' and label == 'O':
            n_errors += 1

            # Count the feature values:
            # Are there certain values that cause a lot of errors?
            for feature, value in zip(fn_dict, components):
                fp_dict[feature][value] += 1

        elif prediction == 'O' and label != 'O':
            n_errors += 1

            # Count the feature values:
            # Are there certain values that cause a lot of errors?
            for feature, value in zip(fn_dict, components):
                fn_dict[feature][value] += 1

    print('Number of errors:', n_errors)
    print('Number of samples:', n_samples)
    print(round(n_errors/n_samples * 100, 3), '%')
    for feature in fn_dict:
        if feature not in ['lemma', 'prev_token', 'next_token', 'n_grams']:
            print(feature)
            print('False positives:')
            print(fp_dict[feature])
            print('False negatives:')
            print(fn_dict[feature])


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
