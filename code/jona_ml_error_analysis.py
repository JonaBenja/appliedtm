from collections import defaultdict
import sys

def main(argv=None):
    if argv is None:
        argv = sys.argv

    outputfile = argv[1]

    features = ['Token', 'Prevtoken', 'POS', 'Prevpos', 'Chunklabel', 'Capitalization']
    error_dict = dict()
    non_error_dict = dict()
    nnp_pos = {'O': 0, 'othercat': 0}
    nnp_prevpos = {'O': 0, 'othercat': 0}

    for feature in features:
        error_dict[feature] = defaultdict(int)
        non_error_dict[feature] = defaultdict(int)

    with open(outputfile, 'r') as inputfile:
        n_samples = 0
        n_errors = 0
        error_lines = dict()
        for line in inputfile:
            n_samples += 1
            line2 = line
            components = line.rstrip('\n').split()
            output = components[-2]
            label = components[-1]
            if output != label:
                n_errors += 1

            for feature, value in zip(non_error_dict, components):
                non_error_dict[feature][value] += 1

                if output != label:
                    error_dict[feature][value] += 1

                    if feature == 'POS' and value == 'NNP':
                        if label != 'O' and output == 'O':
                            nnp_pos['O'] += 1
                        if label != 'O' and output != 'O':
                            nnp_pos['othercat'] += 1

                    if feature == 'Prevpos' and value == 'NNP':
                        if label != 'O' and output == 'O':
                            nnp_prevpos['O'] += 1
                        if label != 'O' and output != 'O':
                            nnp_prevpos['othercat'] += 1

    print("SAMPLES")
    for error_type in non_error_dict:
        if error_type != 'Token' and error_type != 'Prevtoken':
            for value in non_error_dict[error_type]:
                number = non_error_dict[error_type][value]
                percentage = number / n_samples * 100
                non_error_dict[error_type][value] = (number, round(percentage, 2))

            print(error_type)
            print(non_error_dict[error_type])
            print()

    print("ERRORS")
    for error_type in error_dict:
        if error_type != 'Token' and error_type != 'Prevtoken':
            for value in error_dict[error_type]:
                number = error_dict[error_type][value]
                percentage = number/n_errors*100
                error_dict[error_type][value] = (number, round(percentage, 2))

            print(error_type)
            print([(k, error_dict[error_type][k]) for k in sorted(error_dict[error_type].keys(), key=lambda k: error_dict[error_type][k][0], reverse=True)])
            print()

    print(n_samples, n_errors)
    print(n_errors/n_samples * 100)

    print("NNP POS AND PREVPOS")
    print(nnp_pos)
    print(nnp_prevpos)

if __name__ == '__main__':
    main()

# python error_analysis.py ../data/output/out.SVM_trad_emb.conll