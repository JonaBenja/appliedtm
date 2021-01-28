from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from sklearn.metrics import classification_report
import argparse


def extract_features_and_labels(trainingfile, selected_features):
    '''
    This function extracts features and labels from the training file

    :param trainingfile: the path to the training file
    :param selected_features: the features combination
    :returns: lists of features and annotations
    '''

    data = []
    targets = []
    
    #mapping features to matching columns
    feature_to_index = {'token': 0, 'lemma': 1, 'pos_tag': 2, 'prev_token': 3, 'next_token': 4, 'punctuation': 5, 'affixes': 6, 'n_grams': 7}
    with open(trainingfile, 'r', encoding='utf8') as infile:
        for i,line in enumerate(infile):
            if i == 0:
                pass
            else:
                components = line.rstrip('\n').split()
                #checks if the row exists
                if len(components) > 0:   
                    feature_dict = {}
                    for feature_name in selected_features:
                        components_index = feature_to_index.get(feature_name)
                        feature_dict[feature_name] = components[components_index]
                    data.append(feature_dict)
                    
                    # the gold label is in the last column
                    targets.append(components[-1])
    return data, targets


def create_classifier(train_features, train_targets):
    '''
    This function creates classifiers based on SVM

    :param train_features: the list of training features
    :param train_targets: the list of training annotations
    :returns: the model and the vectors
    '''

    #running linear SVC model 
    model = LinearSVC()
    
    #initialize vector transformer
    vec = DictVectorizer()
    
    #transform features 
    features_vectorized = vec.fit_transform(train_features)
    
    #fit model
    model.fit(features_vectorized, train_targets)

    return model, vec



def get_predicted_and_gold_labels(testfile, vectorizer, classifier, selected_features):
    '''
    Function that extracts features and runs the classifier on a test file returning predicted and gold labels

    :param testfile: path to the (preprocessed) test file
    :param vectorizer: vectorizer in which the mapping between feature values and dimensions is stored
    :param classifier: the trained classifier
    :param selected_features: the features combination

    :return predictions: list of output labels provided by the classifier on the test file
    :return goldlabels: list of gold labels as included in the test file
    '''

    # we use the same function as above (guarantees features have the same name and form)
    features, goldlabels = extract_features_and_labels(testfile, selected_features)
    
    # we need to use the same fitting as before, so now we only transform the current features according to this mapping (using only transform)
    test_features_vectorized = vectorizer.transform(features)
    predictions = classifier.predict(test_features_vectorized)

    return predictions, goldlabels




def print_confusion_matrix(predictions, goldlabels):
    '''
    Function that prints out a confusion matrix

    :param predictions: predicted labels
    :param goldlabels: gold standard labels
    :type predictions, goldlabels: list of strings

    :returns: confusion matrix
    '''

    # based on example from https://datatofish.com/confusion-matrix-python/
    data = {'Gold': goldlabels, 'Predicted': predictions}
    df = pd.DataFrame(data, columns=['Gold', 'Predicted'])

    confusion_matrix = pd.crosstab(df['Gold'], df['Predicted'], rownames=['Gold'], colnames=['Predicted'])
    print(confusion_matrix)
    return confusion_matrix


def print_precision_recall_fscore(predictions, goldlabels):
    '''
    Function that prints out precision, recall and f-score in a complete report

    :param predictions: predicted output by classifier
    :param goldlabels: original gold labels
    :type predictions, goldlabels: list of strings
    '''

    report = classification_report(goldlabels,predictions,digits = 3)

    print('METRICS: ')
    print()
    print(report)


def run_classifier(trainfile, testfile, selected_features):
    '''
    Function that runs the classifier and prints the evaluation

    :param trainfile: path to the training file
    :param testfile: path to the test file
    :return: predictions
    '''

    #evaluation of the performances of the SVM with the classical features
    modelname = 'SVM'
    feature_values, labels = extract_features_and_labels(trainfile, selected_features)
    classifier, vectorizer = create_classifier(feature_values, labels)
    predictions, goldlabels = get_predicted_and_gold_labels(testfile, vectorizer, classifier, selected_features)
    print()
    print('---->'+ modelname + ' with ' + ' and '.join(selected_features) + ' as features <----')
    print_precision_recall_fscore(predictions, goldlabels)
    print('------')
    
    return predictions

def main():

    parser = argparse.ArgumentParser(description= 'This script trains two classifiers based on SVM.')

    parser.add_argument('trainfile', help='file path to training data with the new features. Recommended path: "../data/SEM-2012-SharedTask-CD-SCO-training-simple-preprocessed-features.conll"')
    parser.add_argument('testfile', help='file path to the test data with the new features. Recommended path: "../data/SEM-2012-SharedTask-CD-SCO-dev-simple-preprocessed-features.conll"')

    args = parser.parse_args()
    
    #traditional features (SVM)
    selected_features_traditional = ["token", "lemma","pos_tag","prev_token","next_token","punctuation"]
    run_classifier(args.trainfile, args.testfile, selected_features_traditional) 
    
    #with morphological features (SVM-MORPH)
    selected_features_morph = ["token", "lemma","pos_tag","prev_token","next_token","punctuation", "affixes", "n_grams"]
    
    #saving predictions to variable 
    predictions = run_classifier(args.trainfile, args.testfile, selected_features_morph)
    
    #writing best performing system with predictions (SVM-MORPH)
    test = pd.read_csv(args.testfile, encoding = 'utf-8', sep = '\t')
    test['prediction'] = predictions
    filename = args.testfile.replace(".conll", "-prediction.conll")
    test.to_csv(filename, sep = '\t', index = False)
    
        

if __name__ == '__main__':
    main()


