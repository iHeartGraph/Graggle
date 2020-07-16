import sys
import json 
import csv 

from cluster import score, kmeans
from test_arff import get_vectors

LABIC = '/mnt/raid0_24TB/datasets/ContextGraph/'
    
def multi_clusters(fname, ns, thresh=5, nw=100, wl=10, w2v={}, ignore_list=[],top_n=None):
    X,y = get_vectors(
        fname, 
        thresh=thresh, 
        nw=nw, 
        wl=wl, 
        w2v_params=w2v, 
        top_n=top_n,
        ignore_list=ignore_list
    )
    
    for n in ns:
        print('N=%d' % n)
        print(json.dumps(score(X,y,n)[0], indent=4))
        print()
   
class EmptyStdout():
    def __init__(self):
        pass 
    def flush(self):
        pass 
    def write(self, *params):
        pass
     

def thresh_search(fname, nw=200, wl=10, w2v={}, ignore_list=[],top_n=None): 
    f = open('thresh_search.csv', 'w+')
    cw = csv.writer(f)
    cw.writerow(['Thresh', 'Purity'])
    
    og = sys.stdout 
    be_quiet = EmptyStdout()
    
    for t in range(0,255):
        t = t/5
        purities = []
        
        sys.stdout = be_quiet
        for _ in range(10):
            X,y = get_vectors(
                fname, 
                thresh=t, 
                nw=nw, 
                wl=10, 
                w2v_params=w2v, 
                top_n=top_n,
                ignore_list=ignore_list
            )
    
            n = len(set(y))+1
            purities.append(
                kmeans(X,y,n)['purity']['Total']
            )
            
        sys.stdout = og 
        purity = sum(purities)/5
        print('T:%f, Purity: %0.4f' % (t, purity))
        cw.writerow([t, purity])
    
    f.close()
    
def wl_search(fname, ignore_list=[], top_n=None, thresh=5, w2v={}):
    f = open('wl_search.csv', 'w+')
    cw = csv.writer(f)
    cw.writerow(['Num Walks', 'Walk Len', 'Purity'])
    
    og = sys.stdout 
    be_quiet = EmptyStdout()
    
    for nw in range(100, 900, 100):
        if nw == 0:
            nw = 10
        
        for window in range(2, 10):
            w2v['window'] = window
            
            purities = []
            
            sys.stdout = be_quiet
            for _ in range(5):
                X,y = get_vectors(
                    fname, 
                    thresh=thresh, 
                    nw=nw, 
                    wl=10, 
                    w2v_params=w2v, 
                    top_n=top_n,
                    ignore_list=ignore_list
                )
        
                n = len(set(y))+1
                purities.append(
                    kmeans(X,y,n)['purity']['Total']
                )
                
            sys.stdout = og 
            purity = sum(purities)/5
            print('NW: %d, WL:%d, Purity: %0.4f' % (nw, window, purity))
            cw.writerow([nw, window, purity])
    
    f.close()

# This is clearly different data than is used by the other
# benchmark, and they didn't make it easy to find thier 
# dataset...
def r8_multi():
    multi_clusters(
        LABIC + 'Reuters-21578.arff',
        range(8, 33, 8),
        thresh=5, wl=1, nw=800, top_n=8, 
        w2v=dict(
            size=512,
            negative=75,
            window=3
        )
    )
    
# Best params
def webkb_multi():
    multi_clusters(
        LABIC + 'webkb.arff',
        range(4, 17, 4),
        thresh=5, wl=1, nw=800, top_n=4, 
        w2v=dict(
            size=512,
            negative=50,
            window=2
        ),
        ignore_list=['"other"']
    )
    
def cstr_search():
    wl_search(
        LABIC + 'CSTR.arff',
        w2v=dict(
            size=128,
            negative=5,
            window=3
        ),
        thresh=3.4
    )
    
cstr_search()
