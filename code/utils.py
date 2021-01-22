import csv
import nltk
from nltk.stem import WordNetLemmatizer


def wordnet_pos(pos):
    """
    Converts the PoS tags in WordNet readable tags
    :param pos: nltk Pos tag
    :return: WordNet tag
    """
    if pos.startswith('J'):
        return 'a'
    elif pos.startswith('V'):
        return 'v'
    elif pos.startswith('N'):
        return 'n'
    elif pos.startswith('R'):
        return 'r'
    else:
        return 'n'

def lemma_extraction(tokens, pos_list):
    """
    Function to extract lemmas from tokens.
    """
    lemmas = []
    lem = WordNetLemmatizer()
    for token, pos in zip(tokens, pos_list):
        lemma = lem.lemmatize(token, wordnet_pos(pos))
        lemmas.append(lemma)
        
    return lemmas
    
def pos_extraction(tokens):
    """
    Function to extract part-of-speech tags from tokens.
    """
    pos_list = []
    tagged = nltk.pos_tag(tokens)
    for token, pos_tag in tagged:
        pos_list.append(pos_tag)   
        
    return pos_list

def previous_and_next_token_extraction(tokens):
    """
    Function to extract previous and preceding token from tokens list.
    """
    position_index = 0

    prev_tokens = []
    next_tokens = []
    
    for i in range(len(tokens)):

        prev_index = (position_index - 1)
        next_index = (position_index + 1)
        
        #previous token
        if prev_index < 0:
            previous_token = "None"
        else: 
            previous_token = tokens[prev_index]

        prev_tokens.append(previous_token)
            
        #next token
        if next_index < len(tokens):
            next_token = tokens[next_index]
        else: 
            next_token = "None"

        next_tokens.append(next_token)
            
        #moving to next token in list 
        position_index += 1
    
    return prev_tokens, next_tokens
                
        
def is_punctuation(tokens):
    """
    Function to determine if a token is a punctuation mark.
    """
    punctuation = []
    for token in tokens:
        
    #assigning 1 when token is punctuation mark    
        if not token.isalnum():
            punct = 1
        else:
            punct = 0
        
        punctuation.append(punct)
    
    return punctuation
