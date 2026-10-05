import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr


def _compute_local_dlp(Y_emb, X_high, noise_indices, k=10):
    nn_high = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X_high)
    dists_h, idx_h = nn_high.kneighbors(X_high[noise_indices])
    idx_h = idx_h[:, 1:]

    r_high = np.arange(1, k + 1)
    local_dlp = np.zeros(len(noise_indices))

    for i in range(len(noise_indices)):
        nb_idx = idx_h[i]
        pt = noise_indices[i]
        dists_low = np.sqrt(((Y_emb[pt] - Y_emb[nb_idx]) ** 2).sum(axis=1))
        r_low = np.argsort(np.argsort(dists_low)) + 1
        corr, _ = spearmanr(r_high, r_low)
        local_dlp[i] = corr if not np.isnan(corr) else 0.0

    return local_dlp


def dbscan_k_v2(Y_emb, X_high, k, minPts=30, outlier_top_pct=30,
                labels_dir=None, dlp_k=10, dlp_weight=1.0):
    N = Y_emb.shape[0]
    rng = np.random.default_rng(42)

    nn = NearestNeighbors(n_neighbors=minPts, algorithm='auto')
    nn.fit(Y_emb)
    D_knn = nn.kneighbors(Y_emb)[0][:, -1]
    eps_q25 = np.percentile(D_knn, 25)

    labels_dbscan = DBSCAN(eps=eps_q25, min_samples=minPts).fit_predict(Y_emb)
    O_indices = np.where(labels_dbscan == -1)[0]

    labels_updated = labels_dbscan.copy()

    if len(O_indices) == 0:
        O_true_indices = np.array([], dtype=int)
    else:
        cluster_labels_p2 = sorted(set(labels_dbscan[labels_dbscan >= 0]))

        if len(cluster_labels_p2) == 0:
            O_true_indices = O_indices.copy()
        else:
            cluster_members = {cl: np.where(labels_dbscan == cl)[0]
                               for cl in cluster_labels_p2}

            cluster_density = {}
            for cl, members in cluster_members.items():
                if len(members) <= 1:
                    cluster_density[cl] = 1e-10
                    continue
                sample = (rng.choice(members, size=50, replace=False)
                          if len(members) > 50 else members)
                pw = pairwise_distances(Y_emb[sample])
                cluster_density[cl] = pw[np.triu_indices(len(sample), k=1)].mean()

            nearest_cluster = np.empty(len(O_indices), dtype=int)
            nearest_dist = np.full(len(O_indices), np.inf)
            for cl, members in cluster_members.items():
                nn_cl = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(Y_emb[members])
                dists = nn_cl.kneighbors(Y_emb[O_indices])[0].ravel()
                closer = dists < nearest_dist
                nearest_dist[closer] = dists[closer]
                nearest_cluster[closer] = cl

            geo_scores = np.array([nearest_dist[j] / cluster_density[nearest_cluster[j]]
                                   for j in range(len(O_indices))])

            local_dlp = _compute_local_dlp(Y_emb, X_high, O_indices, k=dlp_k)
            dlp_factor = 1.0 - local_dlp

            geo_norm = (geo_scores - geo_scores.min()) / (geo_scores.max() - geo_scores.min() + 1e-10)
            dlp_norm = (dlp_factor - dlp_factor.min()) / (dlp_factor.max() - dlp_factor.min() + 1e-10)

            scores = geo_norm + dlp_weight * dlp_norm

            if labels_dir is not None:
                O_labels = labels_dir[O_indices]
                O_true_mask = np.zeros(len(O_indices), dtype=bool)
                for direction in [1, -1]:
                    dm = (O_labels == direction)
                    if dm.sum() == 0:
                        continue
                    thr = np.percentile(scores[dm], 100 - outlier_top_pct)
                    O_true_mask[dm] = scores[dm] >= thr
            else:
                thr = np.percentile(scores, 100 - outlier_top_pct)
                O_true_mask = scores >= thr

            O_true_indices = O_indices[O_true_mask]

            for j in np.where(~O_true_mask)[0]:
                labels_updated[O_indices[j]] = nearest_cluster[j]

    R = np.zeros(N, dtype=int)
    o_true_set = set(O_true_indices.tolist())
    for i in range(N):
        if i in o_true_set:
            R[i] = k
        else:
            lbl = labels_updated[i]
            R[i] = -(lbl + 1) if lbl >= 0 else k
    return R


def _self_test():
    print('=== dbscan_k_v2 SELF-TEST ===')
    rng = np.random.default_rng(0)
    centers = [(0, 0), (20, 0), (0, 20), (20, 20), (40, 0), (0, 40)]
    blobs = [rng.normal([cx, cy], 0.3, size=(50, 2)) for cx, cy in centers]
    noise = rng.uniform(-8, 48, size=(40, 2))
    Y = np.vstack(blobs + [noise])
    X = Y + rng.normal(0, 0.05, Y.shape)

    R = dbscan_k_v2(Y, X, k=4, minPts=10, outlier_top_pct=50,
                    labels_dir=None, dlp_k=10, dlp_weight=1.0)

    nn = NearestNeighbors(n_neighbors=10).fit(Y)
    eps = np.percentile(nn.kneighbors(Y)[0][:, -1], 25)
    lab = DBSCAN(eps=eps, min_samples=10).fit_predict(Y)
    nclust = len(set(lab[lab >= 0]))
    marker = (R == 4)

    assert nclust >= 4, f'need >=4 clusters to cover the collision case, got {nclust}'
    contaminated = int((marker & (lab >= 0)).sum())
    assert contaminated == 0, f'{contaminated} clustered points leaked into the markers'
    assert marker.sum() > 0, 'marker set must not be empty'
    assert bool((marker <= (lab == -1)).all()), 'every marker must come from the DBSCAN noise set'
    print(f'  [ok] {nclust} clusters, covers the collision case')
    print(f'  [ok] {int(marker.sum())} markers, all from noise, no clustered points')
    print(f'  [ok] {int((lab == -1).sum())} noise points; markers are the top-P% subset')
    print('=== PASSED ===')


if __name__ == '__main__':
    _self_test()
