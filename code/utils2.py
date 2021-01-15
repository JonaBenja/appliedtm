import csv
import nltk
from nltk.stem import WordNetLemmatizer


def lemma_extraction(tokens):
    """
    Function to extract lemmas from tokens.
    """
    lemmas = []
    for token in tokens: 
        lem = WordNetLemmatizer() 
        lemma = lem.lemmatize(token)
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
    
    prev_index = (position_index - 1)
    next_index = (position_index + 1)
    
    position_index = 0
    
    for token in tokens: 
        
        prev_tokens = []
        next_tokens = []
        
        #previous token
        if prev_index < 0:
            previous_token = ""
        else: 
            previous_token = tokens[prev_index]
            prev_tokens.append(previous_token)
            
        #next token
        if next_index < (len(tokens)):
            next_token = tokens[next_index]
        else: 
            next_token = "" 
            next_tokens.append(next_token)
            
        #moving to next token in list 
        position_index += 1
    
    return prev_tokens, next_tokens
                
        
def punctuation(tokens):
    """
    Function to determine if a token is a punctuation mark.
    """
    for token in tokens: 
    punctuation = []
    #0 is no punctuation 
    punct = 0
    
    #assigning 1 when token is punctuation mark
    if not token.isalnum():
            punctuation = 1
            
    punctuation.append(punct)
    
    return punctuation
        
