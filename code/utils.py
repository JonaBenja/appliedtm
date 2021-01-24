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

def morphological_rules(tokens):
    """
    Function to check tokens on negational affixes.
    """
    #reading stopwords in as a list 
    stopwords_file = open('stopwords.txt', 'r')
    stopwords = stopwords_file.read()
    stopwords_list = stopwords.split('\n')
    stopwords_file.close()

    greek_roots = ["gn", "mn","ph","pn","rh","th","y","sis","ic"]
    vowels_list = ["e", "a", "o", "i", "u"]

    label_list = []

    for token in tokens: 
        label = 'reg'

        #checking affix a- in combination with Greek roots of the words they appear on
        if len(token) > 2 and token[0] == 'a' and token[1] not in vowels_list and token not in stopwords:
            for item in greek_roots:
                if item in token: 
                    label = 'a'

        #checking affixes in combination with following tokens
        elif len(token) > 3 and token.startswith('il') and token[2] == 'l':
            label = 'il'


        elif len(token) > 3 and token.startswith('im') and (token[2] == 'm' or 'p'):
            label = 'im'


        elif len(token) > 3 and token.startswith('ir') and token[2] == 'r':
            label = 'ir'


        elif len(token) > 5 and token.startswith("un") and not token.startswith("under") and token.endswith(("able", "ible", "ful", "y", "ing")):
            label = 'un'

        #checking occurrences of prefixes and suffixes  
        elif len(token) > 3 and token.startswith('in'):
            label = 'in'


        elif len(token) > 5 and token.startswith(('dis', 'mis', 'non')):
            label = token[0:3]


        elif len(token) > 5 and token.startswith('anti'): 
            label = "anti"


        elif len(token) > 5 and token.startswith('contra'): 
            label = "contra"


        elif len(token) > 5 and token.startswith('counter'): 
            label = "counter"


        elif len(token) > 5 and token.endswith('less'):
            label = 'less'

        elif token in stopwords_list:
            label = 'reg'

        #adding label to the list
        label_list.append(label)

    return label_list
                    
                
    
def creating_ngrams(tokens):
    """
    Function that creates n-gram out of the token when length of token >= 2
    """
    for token in tokens: 
        n_grams = [] 
        if len(token) == 2:
            bigram = [token[i:i+2] for i in range(len(token)-1)]
            n_grams.append(bigram)

        if len(token) == 3:
            trigram = [token[i:i+3] for i in range(len(token)-1)]
            n_grams.append(trigram)

        if len(token) >=4:
            fgram = [token[i:i+4] for i in range(len(token)-1)]
            n_grams.append(fgram)
        
    return n_grams 
