import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def smooth_rank(Y, nidx, tau):
    Y_nb = Y[nidx]
    d2 = ((Y.unsqueeze(1) - Y_nb) ** 2).sum(-1)
    diff = (d2.unsqueeze(2) - d2.unsqueeze(1)) / tau
    diff = diff.clamp(-50, 50)
    sig = torch.sigmoid(diff)
    k = nidx.shape[1]
    eye = torch.eye(k, dtype=torch.bool, device=Y.device).unsqueeze(0)
    sig = sig.masked_fill(eye, 0.0)
    return 1.0 + sig.sum(dim=2)


def spearman_loss(r_high, r_low):
    r_h = r_high - r_high.mean(dim=1, keepdim=True)
    r_l = r_low  - r_low.mean(dim=1, keepdim=True)
    cov = (r_h * r_l).sum(dim=1)
    corr = cov / (r_h.norm(dim=1) * r_l.norm(dim=1) + 1e-8)
    return (1.0 - corr.clamp(-1, 1)).mean()


def compute_DLP_loss_pytorch(Y, nidx, tau):
    n, k = nidx.shape
    r_high = torch.arange(1, k + 1, dtype=Y.dtype, device=Y.device).unsqueeze(0).expand(n, -1)
    r_low = smooth_rank(Y, nidx, tau)
    return spearman_loss(r_high, r_low)


def compute_kl_loss_dense(Y, P_val_t, P_row_t, P_col_t, n, diag_mask):
    D_sq = torch.cdist(Y, Y).pow(2)
    W = 1.0 / (1.0 + D_sq)
    W = W.masked_fill(diag_mask, 0.0)
    Q = W / W.sum()
    return (P_val_t * torch.log(P_val_t / Q[P_row_t, P_col_t].clamp(min=1e-12))).sum()


def run_sr_tsne(Y_init, P_row, P_col, P_val, nidx_np, lambda_dlp,
                n_iter=1000, lr=10.0, tau_start=5.0, tau_end=0.05,
                device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    n = Y_init.shape[0]
    Y = torch.tensor(Y_init.copy(), dtype=torch.float64, device=device, requires_grad=True)
    P_val_t = torch.tensor(P_val, dtype=torch.float64, device=device)
    P_row_t = torch.tensor(P_row, dtype=torch.long, device=device)
    P_col_t = torch.tensor(P_col, dtype=torch.long, device=device)
    nidx = torch.tensor(nidx_np, dtype=torch.long, device=device)
    diag_mask = torch.eye(n, dtype=torch.bool, device=device)
    optimizer = torch.optim.SGD([Y], lr=lr, momentum=0.8)
    if device.type == 'cuda':
        print(f'  [SR-tSNE] using GPU: {torch.cuda.get_device_name(device)}')
    for t in range(1, n_iter + 1):
        tau = tau_start + (tau_end - tau_start) * (t / n_iter)
        optimizer.zero_grad()
        kl = compute_kl_loss_dense(Y, P_val_t, P_row_t, P_col_t, n, diag_mask)
        if lambda_dlp > 0:
            dlp = compute_DLP_loss_pytorch(Y, nidx, tau)
            (kl + lambda_dlp * dlp).backward()
        else:
            kl.backward()
        optimizer.step()
        Y.data -= Y.data.mean(dim=0)
    return Y.detach().cpu().numpy()


def compute_O_DLP(nn_high_indices, Y_low, k=10):
    oh = nn_high_indices[:, :k]
    nn_l = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(Y_low)
    ol = nn_l.kneighbors(return_distance=False)
    return (oh == ol).sum(axis=1) / k


def compute_DLP(nn_high_indices, Y_low, k=10):
    oh = nn_high_indices[:, :k]
    nn_l = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(Y_low)
    ol = nn_l.kneighbors(return_distance=False)
    n = oh.shape[0]
    return np.array([len(set(oh[i]) & set(ol[i])) / k for i in range(n)])
