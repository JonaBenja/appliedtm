import argparse

def load_data(input_file):
    """
    Function that loads the data.
    :param input_file: path to the data file
    :type input_file: string
    :returns: List of tokens and list of labels
    """

    tokens_list = []
    labels_list = []
    with open(input_file, 'r', encoding='utf-8') as infile:

        for line in infile:
            components = line.rstrip('\n').split()

            if len(components) > 0:
                token = components[3]
                label = components[-1]
                tokens_list.append(token)
                labels_list.append(label)

    return tokens_list, labels_list

def write_out(input_file):
    """
    Function that writes the preprocessed data to an output file.
    :param input_file: path to the data file
    :type input_file: string
    """
    tokens_list, labels_list = load_data(input_file)
    output_file = input_file.replace('simple.txt', 'preprocessed.conll')
    with open(output_file, 'w', encoding='utf-8') as f:
        for token, label in zip(tokens_list, labels_list):
            output = '\t'.join([token.lower(), label])
            f.write(output + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file',
                        help='file path to the input data to preprocess. Example path: "../data/SEM-2012-SharedTask-CD-SCO-dev-simple.txt"')
    args = parser.parse_args()

    write_out(args.input_file)

if __name__ == '__main__':
    main()
