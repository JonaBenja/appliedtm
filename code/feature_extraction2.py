import argparse
import nltk
"""
'../data/SEM-2012-SharedTask-CD-SCO-training-preprocessed.tsv'
'../data/SEM-2012-SharedTask-CD-SCO-dev-preprocessed.tsv'
"""


def extract_features(input_file, outputfile):
    """
    This function extracts the PoS, preceding token and preceding PoS and writes the corresponding columns on the outputfile
    :param input_file: original data file
    :param outputfile: outputfile
    :return: new outputfile
    """
    tokens_list = []
    pos_list = []

    #extract the token
    with open(input_file, 'r', encoding='utf8') as infile:

        for line in infile:
            components = line.rstrip('\n').split()

            if len(components) > 0:
                token = components[0]
                tokens_list.append(token)

    #extract the pos, unpack the tuple of (token, pos) and append the pos to a list
    pos_tuple = nltk.pos_tag(tokens_list)
    [pos_list.append(p) for t, p in pos_tuple]

    #open outfile to write the features
    outfile = open(outputfile, 'w')

    position_index = 0

    for line in open(input_file, 'r', encoding='utf8'):

        if len(line.strip('\n').split()) > 0:

            prev_index = (position_index - 1)
            #writes out the PoS, preceding token and preceding pos
            if prev_index < 0:
                outfile.write(line.rstrip('\n') + '\t' + pos_list[position_index] + '\t' +
                              'None' + '\t' + 'None' + '\n')
                position_index += 1

            else:
                outfile.write(line.rstrip('\n') + '\t' + pos_list[position_index] + '\t' +
                              tokens_list[prev_index] + '\t' +
                              pos_list[prev_index] + '\n')
                position_index += 1

    outfile.close()
    return outputfile


# def write_features(input_file):
#     """
#     Function that adds features to the preprocessed data and writes it to an output file.
#     :param input_file: path to the preprocessed data file
#     :type input_file: string
#     """
#     # Prepare output file
#     output_file = input_file.replace('.tsv', '-features.tsv')
#     outfile = open(output_file, 'w', encoding='utf-8')
#
#     # Read in preprocessed file
#     with open(input_file, 'r', encoding='utf-8') as infile:
#
#         # Extract token and label from preprocessed file
#         for line in infile:
#             components = line.rstrip('\n').split()
#             token = components[0]
#             label = components[-1]
#
#             # Write token to
#             outfile.write(token + '\t')
#
#             # Write lemma feature value
#             lemma = 'lemma'
#             outfile.write(lemma + '\t')
#
#             # Write POS feature value
#             pos = 'POS'
#             outfile.write(pos + '\t')
#
#             # Write NE feature value
#             ne = 'NE'
#             outfile.write(ne + '\t')
#
#             # Append label to line
#             outfile.write(label + '\n')

#    outfile.close()

def main():
    # Set up command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file

    # Write features to the feature file
    # write_features(input_file)
    extract_features(input_file,output_file)

if __name__ == '__main__':
    main()