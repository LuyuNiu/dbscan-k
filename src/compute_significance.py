import pandas as pd
from scipy import stats

COSTS = [0.0, 0.2, 0.5, 1.0, 2.0]

panel = pd.read_csv('results/panel_35.csv').set_index('tag')
STOCKS = list(panel.index)
per = pd.read_csv('results/baselines_perstock35.csv')
per = per[per.ticker.isin(STOCKS)]

def series(method, c):
    if method == 'ours':
        return panel[f'o{c:g}'].values
    if method == 'orig':
        return panel[f'g{c:g}'].values
    sub = per[(per.method == method) & (per.cost_bp == c)].set_index('ticker')
    return sub.loc[STOCKS, 'ret'].values

rows = []
for c in COSTS:
    ours = series('ours', c)
    for name, other in [('B&H', series('B&H', c)),
                        ('orig', series('orig', c)),
                        ('RSI', series('RSI(14)', c))]:
        d = ours - other
        w_stat, w_p = stats.wilcoxon(d, alternative='two-sided')
        t_stat, t_p = stats.ttest_rel(ours, other)
        rows.append(dict(cost_bp=c, vs=name, mean_diff=d.mean(),
                         wins=int((d > 0).sum()), n=len(d),
                         wilcoxon_p=w_p, ttest_p=t_p))

df = pd.DataFrame(rows)
df.to_csv('results/significance_tests.csv', index=False)
pd.set_option('display.float_format', lambda x: f'{x:.3g}')
print(df.to_string(index=False))
print('\nmax Wilcoxon p =', f"{df.wilcoxon_p.max():.2e}",
      '| max t-test p =', f"{df.ttest_p.max():.2e}")
