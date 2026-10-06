import sys, numpy as np, pandas as pd, collections
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
g=pd.read_csv('course_genre.csv'); g.columns=[c.replace('﻿','') for c in g.columns]
prof=pd.read_csv('user_profile.csv'); test=pd.read_csv('rs_content_test.csv'); ratings=pd.read_csv('ratings.csv')
title=g.set_index('COURSE_ID').TITLE
genres=g.columns[2:]
users=test.user.unique(); print('test users', len(users), 'test rows', len(test))
enrolled=test.groupby('user').item.apply(set).to_dict()
all_courses=set(g.COURSE_ID)
def report(name, recs):
    counts=[len(v) for v in recs.values()]
    c=collections.Counter(i for v in recs.values() for i in v)
    print(f"== {name}: avg new courses/user {np.mean(counts):.2f}, users with >=1 rec {np.mean([n>0 for n in counts]):.3f}")
    print('top10', [(i, title.get(i,'?'), n) for i,n in c.most_common(10)])
# 1 profile x genre
P=prof.set_index('user'); G=g.set_index('COURSE_ID')[genres]
for thr in [10.0, 20.0, 30.0]:
    recs={}
    for u in users:
        s=G.values@P.loc[u,genres].values.astype(float)
        sc=pd.Series(s,index=G.index); sc=sc[~sc.index.isin(enrolled[u])]
        recs[u]=list(sc[sc>=thr].sort_values(ascending=False).index)
    report(f'profile thr={thr}', recs)
# 2 similarity
sim=pd.read_csv('sim.csv').values
bows=pd.read_csv('courses_bows.csv'); idx=bows[['doc_index','doc_id']].drop_duplicates().set_index('doc_id').doc_index.to_dict(); rid={v:k for k,v in idx.items()}
for thr in [0.4, 0.5, 0.6]:
    recs={}
    for u in users:
        e=[idx[c] for c in enrolled[u] if c in idx]
        if not e: recs[u]=[]; continue
        m=sim[e].max(axis=0)
        recs[u]=[rid[j] for j in np.where(m>=thr)[0] if rid[j] not in enrolled[u]]
    report(f'similarity thr={thr}', recs)
# 3 clustering
X=StandardScaler().fit_transform(P[genres].values)
for k in range(2,31,2):
    print('k',k,'inertia',round(KMeans(k,n_init=5,random_state=42).fit(X).inertia_))
pca=PCA().fit(X); cum=np.cumsum(pca.explained_variance_ratio_); print('pca cum', cum.round(3).tolist())
npc=int(np.searchsorted(cum,0.9)+1); print('npc90',npc)
K=int(sys.argv[1]) if len(sys.argv)>1 else 20
for name,Z in [('raw',X),('pca',PCA(npc,random_state=42).fit_transform(X))]:
    lab=pd.Series(KMeans(K,n_init=10,random_state=42).fit_predict(Z),index=P.index)
    r=ratings[ratings.user.isin(lab.index)].copy(); r['c']=r.user.map(lab)
    pop=r.groupby(['c','item']).size()
    for minc in [10, 100]:
        recs={}
        for u in users:
            if u not in lab.index: recs[u]=[]; continue
            cp=pop.loc[lab[u]]; recs[u]=[i for i in cp[cp>=minc].index if i not in enrolled[u]]
        report(f'cluster {name} k={K} min_enroll={minc}', recs)
