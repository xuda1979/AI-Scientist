import torch
import networkx as nx

def center(K):
    n = K.size(0)
    H = torch.eye(n, device=K.device) - torch.ones((n,n), device=K.device)/n
    return H @ K @ H

def normalize_unit_diag(K, eps=1e-8):
    K = 0.5*(K + K.T)
    d = torch.diag(K).clamp(min=eps)
    D_inv = torch.diag(1.0 / torch.sqrt(d))
    return D_inv @ K @ D_inv

def psd_softplus(K, beta=100.0):
    K = 0.5*(K + K.T)
    evals, evecs = torch.linalg.eigh(K)
    evals_sp = torch.nn.functional.softplus(beta*evals)/beta
    return (evecs * evals_sp) @ evecs.T

def mmd2_centered(K_teacher, K_student):
    diff = center(K_teacher) - center(K_student)
    return torch.mean(diff**2)

def distill_loss(K_teacher, K_student, Y, lam_diag=1e-3, lam_align=1e-3):
    loss_mmd = mmd2_centered(K_teacher, K_student)
    loss_diag = torch.max(torch.abs(torch.diag(K_teacher) - torch.diag(K_student)))**2
    Ky = center(K_student)
    YY = center(Y @ Y.T)
    align = torch.sum(Ky*YY) / (torch.linalg.norm(Ky)*torch.linalg.norm(YY) + 1e-12)
    return loss_mmd + lam_diag*loss_diag + lam_align*(1.0 - align)

def edge_color_matchings(G: nx.Graph):
    # Greedy/Vizing-inspired heuristic for edge-coloring into matchings
    colors = {}
    color_index = 0
    for u, v in G.edges():
        assigned = False
        for c in range(color_index):
            # check if color c is free at u and v
            if all(colors.get((min(u,w), max(u,w)), None) != c for w in G.neighbors(u)) and \
               all(colors.get((min(v,w), max(v,w)), None) != c for w in G.neighbors(v)):
                colors[(min(u,v), max(u,v))] = c
                assigned = True
                break
        if not assigned:
            colors[(min(u,v), max(u,v))] = color_index
            color_index += 1
    matchings = [[] for _ in range(color_index)]
    for (u,v), c in colors.items():
        matchings[c].append((u,v))
    return matchings