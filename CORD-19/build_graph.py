import os
import json 
import tqdm
import pickle
import random
import bisect
import numpy as np
import pandas as pd

from math import log
from cord_globals import *
from scipy.sparse import csr_matrix, save_npz 

TF_IDF_THRESHOLD = 15
NUM_DOCS = None

def tf_idf(tf, doc_count):
    idf = log(NUM_DOCS/doc_count)
    return tf*idf

def build_graph():
    # Undirected, regular old graph
    df = pd.read_pickle(CLEAN_META)
    corpus = pickle.load(open(CORPUS_F, 'rb'))
    
    global NUM_DOCS
    NUM_DOCS = len(df)
    
    # Represent graph as a sparse CSR matrix as row slices are important
    # but most papers have very few neighbors
    row = [0]
    cols = []
    data = []
    
    last_idx = 0
    progress = tqdm.tqdm(total=len(df), desc='Number of nodes added:')
    for node_id, _ in df.iterrows():
        doc_dict = pickle.load(open(DICTS+str(node_id), 'rb'))
        col = []
        cdata = []
        col_lookup = {}
        col_idx = 0
        
        # Link with all papers that share significant words
        for word, count in doc_dict.items():
            thresh = tf_idf(count, len(corpus[word]['papers']))
            
            if thresh > TF_IDF_THRESHOLD:
                for neigh_id in corpus[word]['papers']: 
                    # Edge weights are the sum of each tf-idf score of shared words
                    # This is functionally equivilant to using a multi-graph
                    # as later on, we do random walks based on these weights
                    # so P(B|A) is the same in both cases
                    if neigh_id in col_lookup:
                        cdata[col_lookup[neigh_id]] += thresh
                    else:
                        col.append(neigh_id)
                        cdata.append(thresh)
                        col_lookup[neigh_id] = col_idx
                        col_idx += 1
        
        # Update CSR Matrix stuff           
        last_idx += len(col)     
        row.append(last_idx)
        cols += col
        data += cdata
        
        progress.update()
    
    print("Saving matrices")
    np.save(GRAPH_FILES[0], data)
    np.save(GRAPH_FILES[1], cols)
    np.save(GRAPH_FILES[2], row)
    
def run():
    return build_graph()

if __name__ == '__main__':
    run()