import argparse

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
    outfile = open(output_file, 'w', encoding='utf-8')

    # Read in preprocessed file
    with open(input_file, 'r', encoding='utf-8') as infile:

        # Extract token and label from preprocessed file
        for line in infile:
            components = line.rstrip('\n').split()
            token = components[0]
            label = components[-1]

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

    outfile.close()

def main():
    # Set up command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    args = parser.parse_args()
    input_file = args.input_file

    # Write features to the feature file
    write_features(input_file)

if __name__ == '__main__':
    main()