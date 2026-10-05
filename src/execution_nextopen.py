import sys
sys.path.insert(0, 'src')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import insample_fixP as F

PSTAR = {0: 80, 0.2: 80, 0.5: 50, 1: 40, 2: 10}
COSTS = [0, 0.2, 0.5, 1, 2]
STOCKS = ['MGM', 'UAA', 'BX', 'TKO', 'TSN', 'TPR', 'RHI', 'KFY', 'MCHP', 'PCTY',
          'BCPC', 'DTM', 'HSIC', 'THO', 'COO', 'WAL', 'ESNT', 'JLL', 'USFD', 'EIX',
          'TALO', 'MGEE', 'EXP', 'CUBE', 'EME', 'SWX', 'AAP', 'OMC', 'WFRD', 'UTHR',
          'EXR', 'HE', 'POR', 'TRUP', 'PHIN']


def load(tag):
    df = pd.read_csv(f'data/{tag}_n.csv')
    y = np.where(df.marketClose >= df.marketOpen, 1, -1)
    cols = [c for c in df.columns if c != 'Date']
    X = StandardScaler().fit_transform(df[cols].values.astype(np.float64))
    prices = df.marketClose.values.astype(float)
    dates = pd.to_datetime(df.Date)
    Y_er = np.load(f'cache/{tag}_srtsne_lam1.0.npy')
    return X, y, prices, dates, Y_er


def run_nextopen(up_mask, down_mask, closes, opens, cost_bps=0):
    cost_rate = cost_bps / 10000 / 2
    cash, pos, state, pending = 10000.0, 0.0, 'C', None
    for i in range(len(closes)):
        if pending == 'B' and state == 'C' and cash > 0:
            cash -= cash * cost_rate; pos = cash / opens[i]; cash = 0; state = 'P'
        elif pending == 'S' and state == 'P' and pos > 0:
            cash = pos * opens[i]; cash -= cash * cost_rate; pos = 0; state = 'C'
        pending = None
        if down_mask[i] and state == 'C':
            pending = 'B'
        elif up_mask[i] and state == 'P':
            pending = 'S'
    if state == 'P':
        cash = pos * closes[-1]
    return (cash - 10000) / 100


rows = []
for t in STOCKS:
    X, y, closes, dates, Y_er = load(t)
    opens = pd.read_csv(f'data/{t}_n.csv').marketOpen.values.astype(float)
    Y_t = np.load(f'cache/{t}_tsne.npy')
    iu, idn = (y == 1), (y == -1)
    mk_orig = F.dbscan_markers(Y_t)
    bh = (closes[-1] - closes[0]) / closes[0] * 100
    r = {'tag': t, 'bh': bh}
    mk_by_P = {P: (F.dbscan_k_v2(Y_er, X, k=4, minPts=30, outlier_top_pct=P,
                                 labels_dir=y, dlp_k=10, dlp_weight=1.0) == 4)
               for P in set(PSTAR.values())}
    for c in COSTS:
        mk = mk_by_P[PSTAR[c]]
        r[f'ours_no_{c}'] = run_nextopen(mk & iu, mk & idn, closes, opens, c)
        r[f'ours_cl_{c}'] = F.run_trading(mk & iu, mk & idn, closes, c)[0]
        r[f'orig_no_{c}'] = run_nextopen(mk_orig & iu, mk_orig & idn, closes, opens, c)
        r[f'orig_cl_{c}'] = F.run_trading(mk_orig & iu, mk_orig & idn, closes, c)[0]
    rows.append(r)
    print(f'{t:<6} done', flush=True)

df = pd.DataFrame(rows)
df.to_csv('results/nextopen_results.csv', index=False)

print('\n=== mean net return (%), 35 stocks ===')
print(f'{"cost":<7}{"ours close":>11}{"ours n-open":>12}{"orig close":>11}{"orig n-open":>12}{"B&H":>7}')
for c in COSTS:
    print(f'{str(c)+"bp":<7}{df[f"ours_cl_{c}"].mean():>11.2f}{df[f"ours_no_{c}"].mean():>12.2f}'
          f'{df[f"orig_cl_{c}"].mean():>11.2f}{df[f"orig_no_{c}"].mean():>12.2f}{df.bh.mean():>7.2f}')

print('\n=== per-stock wins under next-open fills ===')
for c in COSTS:
    print(f'{c}bp: ours>orig {(df[f"ours_no_{c}"] > df[f"orig_no_{c}"]).sum()}/35   '
          f'ours>B&H {(df[f"ours_no_{c}"] > df.bh).sum()}/35')
