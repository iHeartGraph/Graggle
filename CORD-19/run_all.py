import build_dicts as d
import build_graph as g
import build_embeddings as e

d.run()
gr = g.run()
e.run(g=gr)

# No run method. This is easier
import cluster