import numpy as np 
import nltk
from nltk.corpus import stopwords
import spacy
import click
from tqdm import tqdm
import re


mystopwords = stopwords.words('english') + \
['paper', 'result', 'experiment', 'from', 'subject', 're', 'edu', 'use', 'data', 'method', 
 'based', 'new', 'approach', 'also','system', 'model', 'present', 'research', 'propose', 'base']

def words_only(text, regex = re.compile('[A-Za-z]+')):
    return ' '.join(regex.findall(text))

def lemmatize(text: str, nlp = spacy.load('en_core_web_sm')):
    """
    guarantee that text is a string! otherwise it will fail
    """
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

def  remove_stopwords(text, mystopwords = mystopwords):
    return ' '.join([token for token in text.split() if not token in mystopwords])

def text_preprocessing(idx, text, preprocessed_data):
    text = words_only(text)
    text = lemmatize(text)
    text = remove_stopwords(text)
    preprocessed_data[idx] = text

@click.command()
@click.option('--file_path')
def preprocess_data(file_path):
    abstracts = np.load(file_path, allow_pickle=True)
    preprocessed_data = {}
    for idx, paper in tqdm(abstracts):
        text_preprocessing(idx, paper, preprocessed_data)
    name = file_path.split('/')[-1]
    np.save(f'all_data/preprocessed_{name}', preprocessed_data)

    
if __name__ == '__main__':
    preprocess_data()