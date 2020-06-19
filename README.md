## Graggle Clustering
By representing a text corpus as a graph, where documents are nodes, and shared words between documents are edges, we use node2vec to build vectors that can then be used for unsupervised learning tasks. 

We use this technique on the [CORD-19 dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) to generate node embeddings for papers written about the novel coronavirus SARS-CoV-2 to aid in faster knowledge discovery and present the resulting graph, and cluster labels on the website [https://graphlab.seas.gwu.edu/graggle](https://graphlab.seas.gwu.edu/graggle). 

The code to generate embeddings for the CORD-19 data is available in the [CORD-19 directory](CORD-19/). 

The code that tests Graggle cluster purity against cluster benchmarks is available in the [graggle_benchmarks directory](graggle_benchmarks/)
