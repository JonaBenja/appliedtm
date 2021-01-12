import nltk
import spacy
import pandas
# from utils import *
import argparse


"""
'../data/SEM-2012-SharedTask-CD-SCO-training-simple.txt'
"""

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True)
args = parser.parse_args()

def main():
    file = args.file
    with open(file, 'r') as infile:
        for line in infile:
            print(line)
            break



main()