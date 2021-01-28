import argparse
from collections import defaultdict


def create_feature_dict(feature_list):
    """
    Function to create a feature default dictionary to save count the number of feature values.
    """
    feature_dict = dict()
    for feature in feature_list:
        feature_dict[feature] = defaultdict(int)

    return feature_dict


def error_analysis(inputfile):
    """
    Function to execute an error analysis on the prediction on a test set.
    """
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

            # Skip the header line
            if n_samples > 0:

                # Assign needed features to variables
                token = components[0]
                prev_token = components[3]
                next_token = components[4]
                label = components[-2]
                prediction = components[-1]

                # If they are not the same, the system made an error
                if prediction != label:
                    n_errors += 1
                    error_type = prediction + ' instead of ' + label

                    # Create feature dict for every error type
                    if error_type not in ft_error_dict:
                        feature_dict = create_feature_dict(features)
                        ft_error_dict[error_type] = feature_dict

                    # Count values for each feature per error typr
                    for feature, value in zip(features, components):
                        ft_error_dict[error_type][feature][value] += 1

                    # Save token, prev_token and next_token for each error per error type
                    tok_error_dict[error_type].append([n_samples, prev_token, token, next_token])

            n_samples += 1

    # Print tokens and context for errors per error type
    for error_type in ft_error_dict:
        print('------------------------------------------------------------------------')
        print('-----> ', error_type)
        print(len(tok_error_dict[error_type]), 'error(s) of this type')
        print()
        for error in tok_error_dict[error_type]:
            print('Line number:', error[0])
            print('Preceding token:   ', error[1])
            print('The error:         ', error[2])
            print('Following token:   ', error[3])
            print()

        # Print statistics on feature values per error type
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
                             'Example path: "../data/SEM-2012-SharedTask-CD-SCO-dev-simple-features-prediction.conll"')

    args = parser.parse_args()
    input_file = args.input_file

    # Perform error analysis
    with open(input_file, 'r') as infile:
        error_analysis(infile)


if __name__ == '__main__':
    main()
