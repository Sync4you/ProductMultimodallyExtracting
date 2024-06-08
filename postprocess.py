import pandas as pd

df_lyk = pd.read_pickle('submission_lyak.pkl')
df_lyk_gcn = pd.read_pickle('submission_lyak_gcn.pkl')
df_tkm = pd.read_pickle('submission_tkm.pkl')

df_lyk['pred'] *= 1
df_lyk_gcn['pred'] *= 3
df_tkm['pred'] *= 2
df_pred = pd.concat([df_lyk, df_lyk_gcn, df_tkm], axis=0, ignore_index=True).groupby(['posting_id', 'posting_id_target'])[['pred']].sum() / 6

df_pred.reset_index(inplace=True)
df_pred.loc[df_pred['posting_id'] == df_pred['posting_id_target'], 'pred'] = 0.5
df_pred.set_index(['posting_id', 'posting_id_target'], inplace=True)

df_pred = df_pred.query('pred > 0')
df_pred = df_pred[df_pred.apply(lambda row: (row.name[1], row.name[0]) in df_pred.index, axis=1)].reset_index()

print(df_pred)

import networkx as nx
from tqdm import tqdm
from cugraph.centrality.betweenness_centrality import edge_betweenness_centrality

G = nx.Graph()
for i, j, w in df_pred[['posting_id', 'posting_id_target', 'pred']].values:
    G.add_edge(i, j, weight=w)

list_remove_edges = []
list_add_edges = []
def split_graph(G):
    list_comp = list(nx.connected_components(G))
    n = len(G.nodes)
    if len(list_comp) == 1:
        map_bet = edge_betweenness_centrality(G, normalized=True)
        map_bet = {(i, j): w  for (i, j), w in map_bet.items() 
                   if G[i][j]['weight'] < 0.15780210284453428}
        if len(map_bet) == 0:
            return
        edge, val = max(map_bet.items(), key=lambda x: x[1])
        if val > 0.11766651703447985:
            G.remove_edge(*edge)
            list_remove_edges.append(edge)
            return split_graph(G)
    else:
        iters = list_comp
        for comp in iters:
            if len(comp) > 6:
                split_graph(nx.Graph(G.subgraph(comp)))
                
split_graph(G)
for edge in list_remove_edges:
    G.remove_edge(*edge)

def get_score(i, j):
    try:
        return G[i][j]['weight']
    except KeyError:
        return -1

posting_ids = df_pred['posting_id'].unique()
matches = []

for i in posting_ids:
    if i in G:
        m = list(set([i] + list(G.neighbors(i))))
    else:
        m = [i]
    if len(m) > 51:
        m = sorted(m, key=lambda x: get_score(i, x), reverse=True)[:51]
    matches.append(' '.join(m))
matched = pd.DataFrame(dict(posting_id=posting_ids, matches=matches))

matched.to_csv('submission.csv', index=False)


print(matched)
