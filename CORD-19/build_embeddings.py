import sys
import random
import numpy as np
import networkx as nx

from joblib import Parallel, delayed
from cord_globals import *
from tqdm import tqdm 
from gensim.models import Word2Vec
from scipy.sparse import load_npz

# Model parameters
NUM_WALKS = 100
WALK_LEN = 5

# W2V params
NUM_WORKERS = 8
W2V_PARAMS = {
    'size': 256,
    'workers': NUM_WORKERS,
    'sg': 1,
    'negative': 25,
    'window': 4,
}

def generate_walks(num_walks, walk_len, g, starter):
    '''
    Generate random walks on graph for use in skipgram
    '''
    
    # Allow random walks to be generated in parallel given list of nodes
    # for each worker thread to explore
    walks = []
    
    # Can't do much about nodes that have no neighbors
    if g[starter].data.shape[0] == 0:
        return [[str(starter)]]*num_walks
    
   # Get all the one-hop walks at the same time
    neighbors = g[starter]
    one_hop = random.choices(
        neighbors.indices,
        weights=neighbors.data,
        k=num_walks
    )
    
    one_hop_walks = [[str(starter), str(n)] for n in one_hop]
    
    # Generate the remaining steps one at a time
    for walk in one_hop_walks:
        n = int(walk[1])
        
        # Random walk with weights based on tf-idf score
        for _ in range(walk_len-1):
            
            # Stop walk if hit a dead end
            if g[n].data.shape[0] == 0:
                break
            
            # Pick a node weighted randomly from neighbors
            next_node = random.choices(
                g[n].indices,
                weights=g[n].data
            )[0]  
            
            walk.append(str(next_node))
            n = next_node 
                
        walks.append(walk)
    
    return walks

def generate_walks_parallel(g, walk_len, num_walks, workers=1):
    '''
    Distributes nodes needing embeddings across all CPUs 
    Because this is just many threads reading one datastructure this
    is an embarrasingly parallel task
    '''
    flatten = lambda l : [item for sublist in l for item in sublist]     
        
    print('Executing tasks')
    # Tell each worker to generate walks on a subset of
    # nodes in the graph
    walk_results = Parallel(n_jobs=workers, prefer='processes')(
        delayed(generate_walks)(
            num_walks, 
            walk_len,
            g,
            node
        ) 
        for node in tqdm(range(g.shape[0]), desc='Walks generated:')
    )
    
    return flatten(walk_results)


def embed_walks(walks, params, fname):
    '''
    Sends walks to Word2Vec for embeddings
    '''
    model = Word2Vec(walks, **params)
    model.save(fname)
    return model.wv.vectors

def load_embeddings(fname=NODE_EMBEDDINGS):
    return Word2Vec.load(fname).wv.vectors

def run(g=None):
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    else:
        fname = NODE_EMBEDDINGS
    
    if g == None:
        print('Loading graph')
        g = load_npz(GRAPH_FILE)
    
    print('Generating walks')
    walks = generate_walks_parallel(g, WALK_LEN, NUM_WALKS, workers=NUM_WORKERS)
    
    print('Embedding walks...')
    embed_walks(walks, W2V_PARAMS, fname)

if __name__ == '__main__':
    run()    
    