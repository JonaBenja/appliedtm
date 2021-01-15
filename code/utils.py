import csv
import nltk
import spacy
from nltk.stem import WordNetLemmatizer

tokens_list = []
pos_list = []
lem = WordNetLemmatizer()
vowels_list = ['a', 'e', 'o', 'u', 'i']


def feature_extracting_and_writing(input_file, output_file):
    """
    Function to extract features from a preprocessed file and writes them to an outputfile.
    :param input_file: preprocessed file
    :param output_file: the outputfile for the features 
    """
    with open (output_file, "w") as outfile:
        with open(input_file, 'r', encoding='utf8') as infile:

            #defining header names 
            features = ["token",
                        "prev_token", 
                        "lemma",
                        "pos_tag",
                        "punctuation",
                        "bigrams",
                        "fgrams",
                        "negative_affix",
                       "gold_label"]

            writer = csv.writer(outfile, delimiter = '\t')
            writer.writerow(features)

            prev_token = ""
            punctuation = 0

            for line in infile:
                components = line.rstrip('\n').split()

                if len(components) > 0:

                    #token
                    token = components[0]
                    outfile.write(token + '\t')
                    tokens_list.append(token)
                    tagged = nltk.pos_tag(tokens_list)

                    #prev_token
                    outfile.write(prev_token + '\t')
                    prev_token = token 

                    #lemma
                    lemma = lem.lemmatize(token)
                    outfile.write(lemma + '\t')

                    #pos_tag
                    for token, pos_tag in tagged:
                        pos_list.append(pos_tag)   
                    outfile.write(pos_tag + '\t')

                    #punctuation
                    if not token.isalnum():
                            punctuation == 1
                    outfile.write(str(punctuation) + '\t')

                    #morphemes using character level n-grams
                    if len(token) >= 2:
                        bigram = [token[i:i+2] for i in range(len(token)-1)]
                        outfile.write(str(bigram) + '\t')
                    else: 
                        outfile.write(" " + '\t')  

                    if len(token) >=4:
                        fgram = [token[i:i+4] for i in range(len(token)-1)]
                        outfile.write(str(fgram) + '\t')
                    else: 
                        outfile.write(" "+ '\t')


                    #negative affixes
                    #checking specific requirements for preceding letters
                    if len(token) > 2 and token[0] == 'a' and token[1] not in vowels_list: 
                        label = 'neg_affix'
                    if len(token) > 2 and token.startswith('il') and token[2] == 'l':
                        label = 'neg_affix'
                    if len(token) > 2 and token.startswith('im') and (token[2] == 'm' or 'p'):
                        label = 'neg_affix'
                    if len(token) > 2 and token.startswith('ir') and token[2] == 'r':
                        label = 'neg_affix'

                    #checking occurrences of prefixes and suffixes    
                    if len(token) > 2 and (token.startswith('dis') or token.startswith('non') or token.startswith('mis')):
                        label = 'neg_affix'
                    if len(token) > 2 and (token.startswith('de') or token.startswith('in') or token.startswith('un')):
                        label = 'neg_affix'
                    if token.endswith('less'):
                        label = 'neg_affix'

                    else:
                        label = 'reg'
                        
                    outfile.write(label + '\t')

                    #gold label
                    gold_label = components[-1]
                    outfile.write(gold_label + '\n')

                    

                    
input_file = '../data/SEM-2012-SharedTask-CD-SCO-training-preprocessed.conll'
output_file = '../data/training-features_file.conll'
feature_extracting_and_writing(input_file, output_file)
