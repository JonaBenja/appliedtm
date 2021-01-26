import pandas as pd
import argparse
from collections import defaultdict, Counter


def obtain_counts(goldannotations, machineannotations):
    '''
    This function compares the gold annotations to machine output

    :param goldannotations: the gold annotations
    :param machineannotations: the output annotations of the system in question
    :type goldannotations: the type of the object created in extract_annotations
    :type machineannotations: the type of the object created in extract_annotations

    :returns: a countainer providing the counts for each predicted and gold class pair
    '''

    # TIP on how to get the counts for each class
    # https://stackoverflow.com/questions/49393683/how-to-count-items-in-a-nested-dictionary, last accessed 22.10.2020
    evaluation_counts = defaultdict(Counter)

    for g, m in zip(goldannotations, machineannotations):
        evaluation_counts[g]['gold'] += 1
        evaluation_counts[g][m] += 1

    return evaluation_counts




def provide_confusion_matrix(evaluation_counts):
    '''
    Read in the evaluation counts and provide a confusion matrix for each class

    :param evaluation_counts: a container from which you can obtain the true positives, false positives and false negatives for each class
    :type evaluation_counts: type of object returned by obtain_counts

    :prints out a confusion matrix
    '''

    # TIP: provide_output_tables does something similar, but those tables are assuming one additional nested layer
    #      your solution can thus be a simpler version of the one provided in provide_output_tables below
    confusion_matrix = pd.DataFrame.from_dict({(i): evaluation_counts[i]
                                               for i in evaluation_counts.keys()},
                                              orient='index')
    confusion_matrix = confusion_matrix.fillna(0)  # make sure we don't get NaN values

    return confusion_matrix

def extract_data(inputfile, datacolumn, delimiter='\t'):
    '''
    This function extracts data represented in the conll format from a file

    :param inputfile: the path to the conll file
    :param datacolumn: the name of the column in which the target data is provided
    :param delimiter: optional parameter to overwrite the default delimiter (tab)
    :type inputfile: string
    :type datacolumn: string
    :type delimiter: string
    :returns: the data as a list
    '''
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    conll_input = pd.read_csv(inputfile, sep=delimiter, header=0)
    data = conll_input[datacolumn].tolist()
    return data

def extract_examples(tokens_list, gold_labels, model_predictions):
    '''
    This function extracts examples of good predictions and mistakes for the same tokens sample

    :param tokens_list: list of the tokens in the data set
    :param gold_labels: list of the gold annotations
    :param model_predictions: list of the predictions
    :returns: wrong_results_dict containing the mistakes with their counts and right_results_dict containing
    the right predictions with their counts
    '''

    wrong_dict = defaultdict(Counter)               # mistakes dict
    right_dict = defaultdict(Counter)               # correct predictions dict
    insight_dict = defaultdict(dict)               # dict containing preciding and following tokens of each mistake

    for token, gold, prediction in zip(tokens_list, gold_labels, model_predictions):
        if gold != prediction:
            mistaken_label = prediction + '_instead_of_' + gold
            wrong_dict[mistaken_label][token] += 1

            insight_dict[token]['PRECIDING'] = tokens_list[tokens_list.index(token) - 1]
            insight_dict[token]['FOLLOWING'] = tokens_list[tokens_list.index(token) + 1]

    wrong_results_dict = dict()
    mistakes_list = []

    # create a dictionary with the wrong predictions and print the results
    print('--> MISTAKES EXAMPLES:')
    print()
    for category, outcome in wrong_dict.items():
        most_common_mistakes = outcome.most_common(5)
        wrong_results_dict[category] = most_common_mistakes
        print(category)
        print(outcome.most_common(5))                       #modify the number for a wider overview
        print()
        for mistake, count in most_common_mistakes:         #append the mistaken tokens on a separate list
            mistakes_list.append(mistake)

    # prints some insights about each mistake the mistakes
    print('------------------------------------------------------------------------')
    print('--> INSIGHTS ON PRECIDING AND FOLLOWING TOKENS FOR EACH MOST COMMON MISTAKE:')
    i = 0
    for tok, infos in insight_dict.items():
        if i < 10:                                      # Modify this number to get more examples
            for category, outcome in wrong_dict.items():
                most_common_mistakes = outcome.most_common(5)
                for word, number in most_common_mistakes:
                    if tok == word:
                        print(str(i + 1), tok, infos)
                        i += 1
        else:
            break

    # iterate inside the list to get how many times the mistaken tokens were correctly predicted
    for mistake in mistakes_list:
        for token, gold, prediction in zip(tokens_list, gold_labels, model_predictions):
            if mistake == token and gold == prediction:
                right_label = 'Correct_' + prediction
                right_dict[right_label][token] += 1

    # create a dictionary with the correct predictions and the number of times they were correct and print the results
    right_results_dict = dict()
    print('------------------------------------------------------------------------')
    print('--> CORRECT PREDICTIONS EXAMPLES:')
    print()
    for label, pred in right_dict.items():
        right_results_dict[label] = pred.most_common(8)
        print(label)
        print(pred.most_common(8))                          #print all of them for an overview
        print()

    return wrong_results_dict, right_results_dict



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('SVM_filepath', help='file path to the SVM output. Recommended path: "../data/SEM-2012-SharedTask-CD-SCO-dev-simple-preprocessed-features-prediction1.conll"')
    args = parser.parse_args()

    # extract the data
    gold_annotations = extract_data(args.SVM_filepath, 'gold_label' , delimiter='\t')   # extract the gold annotations
    svm_predictions = extract_data(args.SVM_filepath, 'prediction' , delimiter='\t')     # extract the predictions
    tokens = extract_data(args.SVM_filepath, 'token', delimiter='\t')              # extract the token

    # extract and print some examples
    wrong_examples_dict, right_examples_dict = extract_examples(tokens, gold_annotations, svm_predictions)

    # get the metrics and the confusion matrix to get a closer look at each category
    prediction_counts = obtain_counts(gold_annotations, svm_predictions)
    cm = provide_confusion_matrix(prediction_counts)
    print('-----------------------')
    print('--> CONFUSION MATRIX')
    print(cm)

    # SAVE TO OUTPUT TABLES
    # wrong_examples = provide_confusion_matrix(wrong_examples_dict)
    # right_examples = provide_confusion_matrix(right_examples_dict)
    # outfile1 = "../../data/w_examples.csv"
    # outfile2 = "../../data/r_examples.csv"
    # wrong_examples.to_csv(outfile1)
    # right_examples.to_csv(outfile2)


if __name__ == '__main__':
    main()