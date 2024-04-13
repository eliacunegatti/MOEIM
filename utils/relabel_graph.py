import pandas as pd
import networkx as nx

df = pd.read_csv('graphs/wiki-vote_di.txt', sep=' ')


a = df["source"].to_list()
b = df["target"].to_list()

c = sorted(list(set(a).union(set(b))))
rename = {}
for i in range(len(c)):
    rename[c[i]] = i


df1 = pd.DataFrame()
df1["node1"] = df["source"].map(rename)
df1["node2"] = df["target"].map(rename)
df1.to_csv('graphs/wiki-vote_di_.txt', sep=' ', header=None, index=False)
G = nx.read_edgelist('graphs/wiki-vote_di_.txt', nodetype=int)
print(G)


