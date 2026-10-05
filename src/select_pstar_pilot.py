import pandas as pd

df = pd.read_csv('results/pilot_grid16.csv')
STOCKS = ['AAPL', 'AEO', 'AMZN', 'BAC', 'CPRI', 'GES', 'GOOG', 'GS',
          'HD', 'HSBC', 'JCP', 'JPM', 'MSFT', 'TGT', 'TIF', 'WMT']
COSTS = [0, 0.2, 0.5, 1, 2]
PS = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95]
EMB = 'SR-tSNE'
PAPER = {0: 80, 0.2: 80, 0.5: 50, 1: 40, 2: 10}

kv = {(t, P): df[(df.tag == t) & (df.emb == EMB) & (df.method == 'K_v2') & (df.P == P)].iloc[0]
      for t in STOCKS for P in PS}
bh0 = {t: df[df.tag == t].iloc[0]['buyhold'] for t in STOCKS}


def bh_cost(t, c):
    cr = c / 10000 / 2
    gross = bh0[t] / 100 + 1
    return ((1 - cr) * gross - 1) * 100


def wins(P, c):
    return sum(1 for t in STOCKS if kv[(t, P)][f'ret_{c}bp'] > bh_cost(t, c))


print('=== wins vs buy-and-hold (entry cost charged), SR-tSNE + K_v2, 16 pilot stocks ===')
print('P\\cost ' + ''.join(f'{str(c)+"bp":>8}' for c in COSTS))
for P in PS:
    print(f'{P:>6} ' + ''.join(f'{wins(P, c):>8}' for c in COSTS))

print('\n=== per cost level: argmax-wins P set vs paper schedule ===')
print(f'{"cost":<8}{"argmax P set":<28}{"max wins":<10}{"paper P*":<10}{"paper wins"}')
for c in COSTS:
    w = {P: wins(P, c) for P in PS}
    best = max(w.values())
    argmax = [P for P in PS if w[P] == best]
    print(f'{str(c)+"bp":<8}{str(argmax):<28}{str(best)+"/16":<10}{PAPER[c]:<10}{w[PAPER[c]]}/16')
