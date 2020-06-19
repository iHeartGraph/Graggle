import os
import nltk
import json
import pickle
import random
import pandas as pd

from cord_globals import *
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

LEMMATIZER = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))

def pipeline(text):
    '''
    Processes raw text to remove all stopwords and lemmatize
    '''
    t = nltk.word_tokenize(text)
    t = [LEMMATIZER.lemmatize(w) for w in t]
    t = [w for w in t if w not in STOPWORDS]
    t = [w for w in t if len(w) > 1]
    return t


def run(save=True):    
    ''' 
    Given several documents, generates 1 dict of all words
    used in the whole corpus, and a dict for each document.
    The dicts map the word to its frequency
    '''
    df = pd.read_csv(META)
    
    # Remove rows with missing values
    df = df[df.abstract.notnull()]
    df = df[df.title.notnull()]
    df = df.reset_index(drop=True)
    
    # Only keep columns we actually use
    df = df[['title', 'url', 'journal', 'authors', 'publish_time', 'abstract']]
    df.to_pickle(CLEAN_META)
    
    progress = tqdm(total=len(df), desc='Papers parsed:')
    corpus = {}

    for idx, row in df.iterrows():
        text = row['abstract']
        if text.lower().startswith('abstract'):
            text = text[len('abstract'):]
        
        text += ' '+ row['title']
        text = pipeline(text)
        
        doc_dict = {}
        for word in text:
            # Assume it's already accounted for in corpus
            if word in doc_dict:
                doc_dict[word] += 1
                corpus[word]['count'] += 1

            else:
                doc_dict[word] = 1
                
                # Make sure to add this paper to the corpus to make building
                # the graph eaiser later on
                if word in corpus:
                    corpus[word]['count'] += 1
                    corpus[word]['papers'].add(idx)
                else:
                    corpus[word] = {'count': 1, 'papers': {idx}}

        if save:
            pickle.dump(
                doc_dict, 
                open(DICTS+str(idx), 'wb+'), 
                protocol=pickle.HIGHEST_PROTOCOL
            )
        
        progress.update()

    if save:
        pickle.dump(
            corpus, 
            open(CORPUS_F, 'wb+'),
            protocol=pickle.HIGHEST_PROTOCOL
        )
    
    return corpus
    
if __name__ == '__main__':
    run()