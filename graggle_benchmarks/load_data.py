import numpy as np
import pandas as pd 
import pickle
import sys
import json 
import time 

from arff_stream import ArffStreamer
from simple_csr import SimpleCSR
from build_embeddings import run  
from cluster import score
from math import log 
from tqdm import tqdm 
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.feature_selection import VarianceThreshold

TF_IDF_THRESHOLD = 18

def load_data(fname, top_n=None, ignore_list=[]):
    reader = ArffStreamer()
    
    class_distro = {}
    dicts = []

    for row in tqdm(reader.decode(open(fname, 'r')), desc='Building dicts'):
        category = row.pop('class')
        
        if category in ignore_list:
            continue 
        
        if category in class_distro:
            class_distro[category] += 1
        else:
            class_distro[category] = 1
        
        idx = list(row.keys())
        cnt = [row[i] for i in idx]
        dicts.append({'cnt': cnt, 'idx': idx, 'class': category})
            
    if top_n:
        class_order = [(k,v) for k,v in class_distro.items()]
        class_order.sort(key=lambda x : x[1])
        ignore_list += [c[0] for c in class_order[:-top_n]]
        
    print("not including the following classes:\n" + str(ignore_list))
    nid = 0
    ret_dicts = []
    corpus = {}
    y = []
    n = set()
    
    prog = tqdm(total=len(dicts), desc='Building vocabulary')
    for d in dicts:
        if d['class'] not in ignore_list:
            n.add(d['class'])
            for i in d['idx']:
                if i in corpus:
                    corpus[i].add(nid)
                else:
                    corpus[i] = set([nid])
            
            y.append(d['class'])
            ret_dicts.append(d)
            nid += 1
        
        else:
            del d
            
        prog.update()
        
    y = np.array(y)
    return corpus, ret_dicts, y, len(n)
   
# Loads CLUTO matrix files
def load_mat_data(fname):
    # Load labels
    with open(fname + '.rclass', 'r') as f:
        s = f.read()
        
    # Discard empty last line
    y = s.split('\n')[:-1]
    n = len(set(y))
    
    with open(fname, 'r') as f:
        s = f.read()
        
    # Discard first and last lines
    docs = s.split('\n')[1:-1]
    
    dicts = []
    corpus = {}
    nid = 0
    
    for doc in docs:
        # First character is always a space for some reason
        line = doc.split(' ')[1:]
        
        cnts = []
        idxs = []
        
        for i in range(len(line)//2):
            word = int(line[i*2])
            cnt = int(line[i*2 + 1])
            
            cnts.append(cnt)
            idxs.append(word)
            
            if word in corpus:
                corpus[word].add(nid)
            else:
                corpus[word] = set([nid])
        
        dicts.append({'idx': idxs, 'cnt': cnts, 'class': y[nid]})
        nid += 1
    
    return corpus, dicts, y, n
    
# Defined in build_graph    
NUM_DOCS = None 
def tf_idf(tf, doc_count):
    idf = log(NUM_DOCS/doc_count)
    return tf*idf

def build_graph(corpus, dicts):
    global NUM_DOCS 
    NUM_DOCS = len(dicts)
    
    # Represent graph as a sparse CSR matrix as row slices are important
    # but most papers have very few neighbors
    row = [0]
    cols = []
    data = []
    
    last_idx = 0
    node_id = 0
    progress = tqdm(total=NUM_DOCS, desc='Number of nodes added:')
    for paper in dicts:
        col = []
        cdata = []
        col_lookup = {}
        col_idx = 0
        
        # Link with all papers that share significant words
        for i in range(len(paper['idx'])):
            word =  paper['idx'][i]
            count = paper['cnt'][i]
            thresh = tf_idf(count, len(corpus[word]))
            
            if thresh > TF_IDF_THRESHOLD:
                for neigh_id in corpus[word]: 
                    # Prevent self-loops
                    if neigh_id == node_id:
                        continue
                    
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
        
        # Appending is supposed to be more efficient than 
        # concatenation via += 
        [cols.append(c) for c in col]
        [data.append(d) for d in cdata]
        
        progress.update()
    
    print("Building matrix from parsed data")
    g = SimpleCSR(np.array(data), np.array(cols), np.array(row))
    
    return g
 
def run_context_graph(fname, thresh=None, nw=100, wl=4, w2v_params={}, 
                      top_n=None, ismat=False, 
                      num_class=None, ignore_list=[]):
    global TF_IDF_THRESHOLD
    if thresh != None:
        TF_IDF_THRESHOLD = thresh
     
    preproc_start = time.time()  
    if ismat:
        c,d,y,n = load_mat_data(fname)
    else:
        c,d,y,n = load_data(fname, top_n=top_n, ignore_list=ignore_list)
        
    num_words = len(c)
    g = build_graph(c,d)
    
    start = time.time()
    model = run(g, nw=nw, wl=wl, w2v_params=w2v_params)
    
    idx_order = [int(i) for i in model.wv.index2entity]
    y = np.take(y, idx_order, axis=0)
    X = model.wv.vectors
    
    # Allow overide for number of clusters
    n = n if not num_class else num_class
    results = score(X,y,n=n)
    end=time.time()
    
    print('Time elapsed: ' + str(end-start))
    print('Time elapsed incl. preproc: ' + str(end-preproc_start))
    print('Thresh: ' + str(TF_IDF_THRESHOLD))
    print('NW: ' + str(nw) + ', WL: ' + str(wl))
    print('Num Classes: ' + str(n))
    print('Num Documents: ' + str(len(X)))
    print("Num words: " + str(num_words))
    print(json.dumps(results, indent=4))
    
def get_vectors(fname, thresh=None, nw=100, wl=4, ignore_list=[], w2v_params={}, top_n=None, sparse=False):
    global TF_IDF_THRESHOLD
    if thresh != None:
        TF_IDF_THRESHOLD = thresh
     
    preproc_start = time.time()      
    c,d,y,n = load_data(fname, top_n=top_n, ignore_list=ignore_list)
        
    print("%d documents; %d words" % (len(d), len(c)))
    g = build_graph(c,d)
    
    start = time.time()
    model = run(g, nw=nw, wl=wl, w2v_params=w2v_params)
    
    idx_order = [int(i) for i in model.wv.index2entity]
    y = np.take(y, idx_order, axis=0)
    X = model.wv.vectors
    
    return X, y
    
def get_metadata(fname, sparse=False):
    DATA_DIR = '/mnt/raid0_24TB/datasets/LABIC/'
    
    if sparse:
        corpus, dicts, y, n = load_sparse_data(DATA_DIR + fname)
    else:
        corpus, dicts, y, n = load_data(DATA_DIR + fname)
        
    print("%s:\n\nNum docs: %d\tNum classes: %d\t Num words: %d" % (fname, len(dicts), n, len(corpus)))