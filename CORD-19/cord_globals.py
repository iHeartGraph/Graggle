import os

# File locations
DATA_HOME = '/mnt/raid0_24TB/datasets/CORD-19/'
HOME = os.path.dirname(__file__)
META = DATA_HOME + 'metadata.csv'
CLEAN_META = HOME + 'meta.pkl'
GRAPH_DF = HOME + 'n2v_meta.pkl'

#   Dict data
DICTS = HOME + 'dictionaries/'
if not os.path.exists(DICTS):
    os.mkdir(DICTS)

CORPUS_F = HOME + 'corpus.data'

#   Graph data
GRAPH_FILE = HOME + 'graph.npz'
NODE_EMBEDDINGS = HOME + 'node_embeddings.model'

GRAPH_FILES = [HOME + 'graph_' + f + '.npy' for f in ['data', 'cols', 'row']]


# Added some stopwords specific to journal papers
# or that slipped through NLTK's default list    
CUSTOM_STOPWORDS = [
    "n't", 
    "'m", 
    "'re", 
    "'s", 
    "nt", 
    "may",
    "also",
    "fig",
    "http"
]