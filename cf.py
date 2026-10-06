import numpy as np, pandas as pd, torch, time
from surprise import Dataset, Reader, KNNBasic, NMF, accuracy
from surprise.model_selection import train_test_split as sp_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, balanced_accuracy_score
torch.manual_seed(42); np.random.seed(42)
r = pd.read_csv('ratings.csv')
tr, te = train_test_split(r, test_size=0.3, random_state=42)
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
print('baseline global mean RMSE', round(rmse(te.rating, np.full(len(te), tr.rating.mean())), 4))
reader = Reader(rating_scale=(2, 3))
trainset = Dataset.load_from_df(tr[['user', 'item', 'rating']], reader).build_full_trainset()
testset = list(te[['user', 'item', 'rating']].itertuples(index=False, name=None))
for k in [10, 40, 80]:
    for sim in ['cosine', 'pearson']:
        algo = KNNBasic(k=k, sim_options={'name': sim, 'user_based': False}, verbose=False).fit(trainset)
        print('KNN item', k, sim, round(accuracy.rmse(algo.test(testset), verbose=False), 4), flush=True)
for f in [15, 32, 64]:
    for ep in [20, 50]:
        algo = NMF(n_factors=f, n_epochs=ep, random_state=42).fit(trainset)
        print('NMF', f, ep, round(accuracy.rmse(algo.test(testset), verbose=False), 4), flush=True)
# neural net with embeddings
uidx = {u: i for i, u in enumerate(r.user.unique())}; iidx = {c: i for i, c in enumerate(r.item.unique())}
def enc(d): return torch.tensor(d.user.map(uidx).values), torch.tensor(d.item.map(iidx).values), torch.tensor(d.rating.values, dtype=torch.float32)
class Net(torch.nn.Module):
    def __init__(s, nu, ni, k):
        super().__init__(); s.ue = torch.nn.Embedding(nu, k); s.ie = torch.nn.Embedding(ni, k); s.ub = torch.nn.Embedding(nu, 1); s.ib = torch.nn.Embedding(ni, 1)
        for e in [s.ue, s.ie]: torch.nn.init.normal_(e.weight, 0, 0.05)
        for e in [s.ub, s.ib]: torch.nn.init.zeros_(e.weight)
        s.mu = torch.nn.Parameter(torch.tensor(2.95))
    def forward(s, u, i): return s.mu + (s.ue(u) * s.ie(i)).sum(1) + s.ub(u).squeeze(1) + s.ib(i).squeeze(1)
U, I, Y = enc(tr); Ut, It, Yt = enc(te)
best = None
for k in [16, 32]:
    torch.manual_seed(42); m = Net(len(uidx), len(iidx), k); opt = torch.optim.Adam(m.parameters(), lr=0.003, weight_decay=0)
    for ep in range(10):
        perm = torch.randperm(len(Y))
        for b in range(0, len(Y), 1024):
            ix = perm[b:b + 1024]; p = m(U[ix], I[ix])
            loss = ((p - Y[ix]) ** 2).mean() + 1e-4 * (m.ue(U[ix]).pow(2).sum(1).mean() + m.ie(I[ix]).pow(2).sum(1).mean())
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): e = rmse(Yt.numpy(), m(Ut, It).clamp(2, 3).numpy())
        print('NN k', k, 'epoch', ep + 1, 'test RMSE', round(e, 4), flush=True)
    if best is None or e < best[0]: best = (e, k, m)
print('NN best', round(best[0], 4), 'k', best[1])
m = best[2]
with torch.no_grad(): UE = m.ue.weight.numpy(); IE = m.ie.weight.numpy()
def feats(d): return UE[d.user.map(uidx).values] + IE[d.item.map(iidx).values]
Xtr, Xte = feats(tr), feats(te)
for name, mdl in [('Linear', LinearRegression()), ('Ridge a=1', Ridge(1.0)), ('Lasso a=0.001', Lasso(0.001)), ('RandomForest', RandomForestRegressor(100, max_depth=12, n_jobs=-1, random_state=42))]:
    s = time.time(); mdl.fit(Xtr, tr.rating); print('Reg', name, round(rmse(te.rating, mdl.predict(Xte).clip(2, 3)), 4), flush=True)
ytr = (tr.rating == 3).astype(int); yte = (te.rating == 3).astype(int)
print('majority class acc', round(yte.mean(), 4))
for name, mdl in [('LogReg', LogisticRegression(max_iter=1000)), ('LogReg balanced', LogisticRegression(max_iter=1000, class_weight='balanced')), ('RandomForest balanced', RandomForestClassifier(200, max_depth=12, class_weight='balanced', n_jobs=-1, random_state=42))]:
    mdl.fit(Xtr, ytr); p = mdl.predict(Xte)
    print('Clf', name, 'acc', round(accuracy_score(yte, p), 4), 'bal_acc', round(balanced_accuracy_score(yte, p), 4), 'macroF1', round(f1_score(yte, p, average='macro'), 4), flush=True)
