import numpy as np

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import homogeneity_completeness_v_measure as hcv

from math import log
from scipy.stats import pearsonr, mode
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from nltk.cluster.kmeans import KMeansClusterer

def entropy(y, y_prime):
    all_e = 0
    le = LabelEncoder()
    le.fit(y)
    
    for label in range(y_prime.max()+1):
        # Find all classes contained in cluster, and make arr
        # numeric so bincount works
        actual = y[y_prime == label]
        actual = le.transform(actual)
        
        e = 0
        for cnt in np.bincount(actual):
            if cnt == 0:
                continue 
            
            P_cnt = cnt/actual.shape[0]
            e += cnt * log(P_cnt,2)
            
        all_e += e
    
    q = le.transform(y).max() + 1
    all_e *= -1/(y_prime.shape[0] * log(q,2))
    
    return all_e

def purity(y, y_prime):
    cluster_purity = {}
    total_purity = 0
    for label in range(y_prime.max()+1):
        actual = y[y_prime == label]
        cnt = mode(actual).count[0]
        cluster_purity[label] = cnt / actual.shape[0]
        total_purity += cnt 
        
    cluster_purity['Total'] = total_purity / y.shape[0]
    return cluster_purity

def get_metrics(y, y_prime):
    p = purity(y, y_prime)
    e = entropy(y, y_prime)
    #rand = adjusted_rand_score(y, y_prime)
    #h,c,v = hcv(y, y_prime)
    
    return {
        #'adj_rand': rand,
        #'homogen': h,
        #'complete': c,
        #'v_measure': v, 
        'purity': p,
        'entropy': e
    }
    

def kmeans(X,y,n):
    km = KMeans(
        n_clusters=n, 
        n_jobs=8,
        n_init=100,
        max_iter=500
    ).fit(X)
    y_prime = km.labels_
    
    metrics = get_metrics(y, y_prime)
    metrics['Algorithm'] = 'K-Means'        
    return metrics

def SLINK(X,y,n):
    sl = AgglomerativeClustering(
        n_clusters=n, 
        linkage='single',
        affinity='cosine'
    ).fit(X)
    
    y_prime = sl.labels_
    
    metrics = get_metrics(y, y_prime)
    metrics['Algorithm'] = 'SLINK'        
    return metrics

def CLINK(X,y,n):
    cl = AgglomerativeClustering(
        n_clusters=n, 
        linkage='complete',
        affinity='cosine'
    ).fit(X)
    
    y_prime = cl.labels_

    metrics = get_metrics(y, y_prime)
    metrics['Algorithm'] = 'CLINK'        
    return metrics

def UPGMA(X,y,n):
    up = AgglomerativeClustering(
        n_clusters=n, 
        linkage='average',
        affinity='cosine'
    ).fit(X)
    
    y_prime = up.labels_
    
    metrics = get_metrics(y, y_prime)
    metrics['Algorithm'] = 'UPGMA'        
    return metrics

def score(X,y,n=120):
    return [
        kmeans(X,y,n),
        SLINK(X,y,n),
        CLINK(X,y,n),
        UPGMA(X,y,n)
    ]