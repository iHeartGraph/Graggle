import sys 
from load_data import run_context_graph

DATA = '/mnt/raid0_24TB/datasets/ContextGraph/'

REUTERS = DATA + 'Reuters-21578.arff'
CSTR = DATA + 'CSTR.arff'
WEBKB = DATA + 'webkb.arff'
KDATA = DATA + 'k1a.mat'


# Tests on various datasets from 
# http://sites.labic.icmc.usp.br/text_collections/

def cstr():
    print('\nCSTR')
    w2v = dict(
        size=128,
        negative=5,
        window=3
    )
    run_context_graph(
        CSTR, 
        thresh=1, 
        nw=200, 
        wl=10, 
        w2v_params=w2v
    )
    
def reuters():
    print('\nReuters')
    w2v = dict(
        size=256,
        negative=100,
        window=3
    )
    run_context_graph(
        REUTERS, 
        thresh=5, 
        wl=1, 
        nw=800, 
        top_n=10, 
        w2v_params=w2v
    )

def webkb4():
    print('\nWebKB-4')
    w2v = dict(
        size=512,
        negative=50,
        window=3
    )
    # Best thresh 5 so far
    run_context_graph(
        WEBKB, 
        thresh=5, 
        wl=1, 
        nw=800,
        top_n=4, 
        w2v_params=w2v,
        ignore_list=['"other"'] # Only counts "entity representing" categories in benchmark
    )
    
def webkb():
    print('\nWebKB')
    w2v = dict(
        size=512,
        negative=50,
        window=3
    )
    # Best thresh 10 so far
    run_context_graph(
        WEBKB, 
        thresh=10, 
        wl=1, 
        nw=800,
        w2v_params=w2v
    )
    
def kdata():
    print("\nK-Dataset")
    w2v = dict(
        size=512,
        negative=100,
        window=3
    )
    run_context_graph(
        KDATA, 
        ismat=True, 
        thresh=1, 
        wl=10,
        nw=200, 
        w2v_params=w2v
    )
    
kdata()