import numpy as np
import pandas as pd 

from cord_globals import *
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

NODE_EMBEDDINGS = HOME + 'node_embeddings.model'
print('Loading Embeddings')
model = Word2Vec.load(NODE_EMBEDDINGS)
v = model.wv.vectors
order = [int(i) for i in model.wv.index2entity]
X = np.zeros((max(order)+1, len(v[1])))
X[order] = v

print("Clustering")
y = KMeans(n_clusters=100).fit(X).labels_

# Nicer visuals but takes longer 
print('Using t-SNE')
simplest = TSNE(n_components=2, n_jobs=8)
X = simplest.fit_transform(X)

df = pd.DataFrame({
    'X0': X[:,0],
    'X1': X[:,1],
    'y': y
})

df.to_pickle(GRAPH_DF)