# Auto-generated from LaTeX code blocks; consolidate all simulation here.
# === Begin extracted block 1 ===
#!/usr/bin/env python3
# Reproducible EB-SGD experiments and ablations.
# How to run: python3 simulation.py
# Outputs (current folder):
# - results.txt (logistic, per-epoch means/CIs across 10 seeds + diagnostics + timings)
# - results_summary.csv (final metrics and mean time per step)
# - results_mlp.txt (2-layer MLP nonconvex task, per-epoch means/CIs + diagnostics + timings)
# - ablation_alpha.csv (gain sweep)
# - ablation_W.csv (window sweep)
# - ablation_H.csv (target entropy sweep)
# - grid_lr.csv (small learning-rate grid for fairness)
# - best_grid.csv (best-of-grid summary used by the fairness figure)
# All numbers in the manuscript are read from these files.

import numpy as np, time, json, sys, platform, math

def sigmoid(z):
    "Numerically stable logistic function."
    return 1.0/(1.0+np.exp(-z))

def loss_and_grad_logreg(w, Xb, yb, lam=1e-4):
    "Binary logistic loss and gradient with L2 regularization for a minibatch."
    p = sigmoid(Xb.dot(w))
    eps = 1e-12
    loss = -np.mean(yb*np.log(p+eps)+(1-yb)*np.log(1-p+eps)) + 0.5*lam*np.sum(w*w)
    grad = Xb.T.dot(p-yb)/len(yb) + lam*w
    return loss, grad

def accuracy_logreg(w, X, y):
    "Binary accuracy for logistic regression."
    p = sigmoid(X.dot(w))
    yhat = (p>0.5).astype(np.float64)
    return float((yhat==y).mean())

def relu(x): return np.maximum(0.0, x)

def init_mlp(d, h, k, rng):
    "He initialization for a 2-layer ReLU MLP."
    return {"W1": rng.randn(d,h)*np.sqrt(2.0/d), "b1": np.zeros(h),
            "W2": rng.randn(h,k)*np.sqrt(2.0/h), "b2": np.zeros(k)}

def softmax(z):
    "Row-wise softmax."
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e/np.sum(e, axis=1, keepdims=True)

def mlp_forward(params, X):
    "Forward pass for the 2-layer MLP."
    h = relu(X.dot(params["W1"]) + params["b1"])
    logits = h.dot(params["W2"]) + params["b2"]
    return h, logits

def mlp_loss_and_grad(params, Xb, yb, lam=1e-4):
    "Cross-entropy loss and gradients for the MLP on a minibatch."
    h, logits = mlp_forward(params, Xb)
    P = softmax(logits)
    n = Xb.shape[0]
    y_one = np.zeros_like(P); y_one[np.arange(n), yb] = 1.0
    loss = -np.sum(y_one*np.log(P+1e-12))/n
    loss += 0.5*lam*(np.sum(params["W1"]**2)+np.sum(params["W2"]**2))
    dlogits = (P - y_one)/n
    dW2 = h.T.dot(dlogits) + lam*params["W2"]; db2 = dlogits.sum(axis=0)
    dh = dlogits.dot(params["W2"].T); dh[h<=0] = 0.0
    dW1 = Xb.T.dot(dh) + lam*params["W1"]; db1 = dh.sum(axis=0)
    return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

def mlp_accuracy(params, X, y):
    "Classification accuracy for the MLP."
    _, logits = mlp_forward(params, X)
    yhat = np.argmax(logits, axis=1)
    return float((yhat==y).mean())

def unit(v):
    "Safe normalization to unit vector."
    n = np.linalg.norm(v)+1e-12
    return v/n

def vMF_entropy_from_window(grad_dir_window):
    """
    vMF-based directional entropy proxy from a window of unit vectors in R^d.
    Uses mean resultant length R and a monotone heuristic for kappa(R).
    Returns a squashed value in [0,1] for robustness.
    """
    if len(grad_dir_window)==0: return 0.5*np.log(2*np.pi)
    d = grad_dir_window.shape[1]
    R_vec = grad_dir_window.mean(axis=0); R = float(np.linalg.norm(R_vec))
    R = float(np.clip(R, 1e-8, 0.999999))
    kappa = (R*(d - R**2)) / (1 - R**2)  # monotone heuristic
    Ad = R  # proxy for A_d(kappa)
    logC = (d/2 - 1)*np.log(max(kappa,1e-12)) - (d/2)*np.log(2*np.pi) - kappa
    H = -kappa*Ad - logC
    return float(1.0/(1.0+np.exp(-(H/(d+1e-9)))))  # squashed to [0,1]

def run_logreg_once(seed, method, epochs=20, batch=64, base_eta=0.05, d=20, N_train=2000, N_test=1000,
                    eb_target=0.65, eb_alpha=2.0, W=50, align_target=0.70, align_beta=2.0):
    "Single run of the synthetic logistic regression task with the specified optimizer."
    rng = np.random.RandomState(seed)
    X = rng.randn(N_train + N_test, d)
    w_true = rng.randn(d)
    logits = X.dot(w_true) + 0.5*rng.randn(N_train + N_test)
    probs = sigmoid(logits); y = (probs > 0.5).astype(np.float64)
    X_train, X_test = X[:N_train], X[N_train:]; y_train, y_test = y[:N_train], y[N_train:]
    w = np.zeros(d); m, v = np.zeros(d), np.zeros(d)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    # SGDM state
    vel = np.zeros(d); mu = 0.9
    # RMSprop state
    s2 = np.zeros(d); rho = 0.99
    window = []; eta0 = base_eta
    results = []
    for ep in range(1, epochs+1):
        idx = np.arange(X_train.shape[0]); rng.shuffle(idx)
        Xbatches = np.array_split(X_train[idx], max(1, X_train.shape[0]//batch))
        ybatches = np.array_split(y_train[idx], max(1, y_train.shape[0]//batch))
        eta_eb_epoch, ent_eb_epoch, align_epoch = [], [], []
        t0 = time.perf_counter()
        for it, (Xb, yb) in enumerate(zip(Xbatches, ybatches), start=1):
            loss, g = loss_and_grad_logreg(w, Xb, yb)
            if method=='SGD':
                step = base_eta*g
            elif method=='SGDM':
                vel = mu*vel + g
                step = base_eta*vel
            elif method=='SGDCosine':
                t = (ep-1 + it/len(Xbatches)); T = epochs
                eta = 0.5*eta0*(1 + math.cos(math.pi * t / T))
                step = eta*g
            elif method=='RMSprop':
                s2 = rho*s2 + (1-rho)*(g*g)
                step = (base_eta*g)/(np.sqrt(s2)+eps)
            elif method=='Adam':
                m = beta1*m + (1-beta1)*g; v = beta2*v + (1-beta2)*(g*g)
                mhat = m/(1-beta1**(ep)); vhat = v/(1-beta2**(ep))
                step = (base_eta*mhat)/(np.sqrt(vhat)+eps)
            elif method=='AdamW':
                weight_decay = 1e-4
                m = beta1*m + (1-beta1)*g; v = beta2*v + (1-beta2)*(g*g)
                mhat = m/(1-beta1**(ep)); vhat = v/(1-beta2**(ep))
                step = (base_eta*mhat)/(np.sqrt(vhat)+eps)
                w *= (1 - base_eta*weight_decay)
            elif method=='Align_SGD':
                gn = np.linalg.norm(g)+1e-12; gh = g/gn
                window.append(gh);  window = window[-W:]
                R_vec = np.mean(window, axis=0); R = float(np.linalg.norm(R_vec))
                eta = base_eta*np.exp(align_beta*(R - align_target))
                eta = float(np.clip(eta, 1e-4, 0.2))
                step = eta*g; align_epoch.append(R)
            elif method=='EB_SGD':
                gn = np.linalg.norm(g)+1e-12; gh = g/gn
                window.append(gh); window = window[-W:]
                Hhat = vMF_entropy_from_window(np.array(window))
                eta = base_eta*np.exp(eb_alpha*(eb_target - Hhat))
                eta = float(np.clip(eta, 1e-4, 0.2))
                step = eta*g; eta_eb_epoch.append(eta); ent_eb_epoch.append(float(Hhat))
            else:
                raise ValueError("Unknown method")
            w -= step
        t1 = time.perf_counter()
        step_time = (t1 - t0)/max(1, len(Xbatches))
        tr_loss,_ = loss_and_grad_logreg(w, X_train, y_train)
        acc = accuracy_logreg(w, X_test, y_test)
        eta_mean = float(np.mean(eta_eb_epoch)) if eta_eb_epoch else float('nan')
        ent_mean = float(np.mean(ent_eb_epoch)) if ent_eb_epoch else float('nan')
        align_mean = float(np.mean(align_epoch)) if align_epoch else float('nan')
        results.append((ep, float(tr_loss), float(acc), eta_mean, ent_mean, align_mean, float(step_time)))
    return results

def run_mlp_once(seed, method, epochs=15, batch=128, base_eta=0.05, d=40, h=64, k=4, N_train=4000, N_test=1000,
                 eb_target=0.60, eb_alpha=1.5, W=50):
    "Single run of the synthetic 4-class MLP task with the specified optimizer."
    rng = np.random.RandomState(seed)
    X = rng.randn(N_train+N_test, d)
    centers = rng.randn(k, d)
    y = rng.randint(0, k, size=N_train+N_test)
    X += centers[y] + 0.3*rng.randn(N_train+N_test, d)
    X_train, X_test = X[:N_train], X[N_train:]; y_train, y_test = y[:N_train], y[N_train:]
    params = init_mlp(d, h, k, rng)
    m, v = {n: np.zeros_like(p) for n,p in params.items()}, {n: np.zeros_like(p) for n,p in params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    window = []; results = []
    for ep in range(1, epochs+1):
        idx = np.arange(X_train.shape[0]); np.random.shuffle(idx)
        Xbatches = np.array_split(X_train[idx], max(1, X_train.shape[0]//batch))
        ybatches = np.array_split(y_train[idx], max(1, y_train.shape[0]//batch))
        eta_eb_epoch, ent_eb_epoch = [], []
        t0 = time.perf_counter()
        for it,(Xb,yb) in enumerate(zip(Xbatches,ybatches), start=1):
            loss, g = mlp_loss_and_grad(params, Xb, yb)
            if method=='SGD':
                eta = base_eta
                for name in params: params[name] -= eta*g[name]
            elif method=='Adam':
                for name in params:
                    m[name] = beta1*m[name] + (1-beta1)*g[name]
                    v[name] = beta2*v[name] + (1-beta2)*(g[name]*g[name])
                    mhat = m[name]/(1-beta1**(ep)); vhat = v[name]/(1-beta2**(ep))
                    params[name] -= (base_eta*mhat)/(np.sqrt(vhat)+eps)
            elif method=='EB_SGD':
                flatg = np.concatenate([g["W1"].ravel(), g["b1"].ravel(), g["W2"].ravel(), g["b2"].ravel()])
                gh = unit(flatg)
                window.append(gh); window = window[-W:]
                Hhat = vMF_entropy_from_window(np.array(window))
                eta = base_eta*np.exp(eb_alpha*(eb_target - Hhat))
                eta = float(np.clip(eta, 1e-5, 0.1))
                for name in params: params[name] -= eta*g[name]
                eta_eb_epoch.append(eta); ent_eb_epoch.append(Hhat)
            else:
                raise ValueError("Unknown method")
        t1 = time.perf_counter()
        step_time = (t1 - t0)/max(1, len(Xbatches))
        tr_loss,_ = mlp_loss_and_grad(params, X_train, y_train)
        acc = mlp_accuracy(params, X_test, y_test)
        eta_mean = float(np.mean(eta_eb_epoch)) if eta_eb_epoch else float('nan')
        ent_mean = float(np.mean(ent_eb_epoch)) if ent_eb_epoch else float('nan')
        results.append((ep, float(tr_loss), float(acc), eta_mean, ent_mean, float(step_time)))
    return results

def aggregate_over_seeds_logreg(seeds, methods, epochs=20, d=20, N_train=2000, N_test=1000,
                                eb_target=0.65, eb_alpha=2.0, W=50, align_target=0.70, align_beta=2.0):
    "Aggregate metrics over seeds for the logistic task; compute means, CIs, and diagnostics."
    per_method = {m: [] for m in methods}
    base_etas = {'SGD':0.05, 'SGDM':0.05, 'SGDCosine':0.05, 'RMSprop':0.01, 'Adam':0.02, 'AdamW':0.02, 'Align_SGD':0.05, 'EB_SGD':0.03}
    for s in seeds:
        for m in methods:
            res = run_logreg_once(seed=s, method=m, epochs=epochs, base_eta=base_etas[m], d=d, N_train=N_train, N_test=N_test,
                                  eb_target=eb_target, eb_alpha=eb_alpha, W=W, align_target=align_target, align_beta=align_beta)
            per_method[m].append(res)
    agg = []
    for ep in range(1, epochs+1):
        row = {'epoch': ep}
        for m in methods:
            losses = [per_method[m][i][ep-1][1] for i in range(len(seeds))]
            accs   = [per_method[m][i][ep-1][2] for i in range(len(seeds))]
            tms    = [per_method[m][i][ep-1][6] for i in range(len(seeds))]
            row[f'loss_{m}_mean'] = float(np.mean(losses))
            row[f'loss_{m}_std']  = float(np.std(losses, ddof=1))
            row[f'acc_{m}_mean']  = float(np.mean(accs))
            row[f'acc_{m}_std']   = float(np.std(accs, ddof=1))
            row[f'time_{m}_ms']   = float(1000*np.mean(tms))
            n = len(seeds)
            row[f'loss_{m}_ci'] = float(1.96*row[f'loss_{m}_std']/np.sqrt(n))
            row[f'acc_{m}_ci']  = float(1.96*row[f'acc_{m}_std']/np.sqrt(n))
            if m=='EB_SGD':
                etas = [per_method[m][i][ep-1][3] for i in range(len(seeds))]
                ents = [per_method[m][i][ep-1][4] for i in range(len(seeds))]
                row['eb_eta_mean'] = float(np.mean(etas))
                row['eb_ent_mean'] = float(np.mean(ents))
            if m=='Align_SGD':
                aligns = [per_method[m][i][ep-1][5] for i in range(len(seeds))]
                row['align_metric_mean'] = float(np.mean(aligns))
        agg.append(row)
    summary = {}
    for m in methods:
        summary[m] = {
            'final_train_loss_mean': agg[-1][f'loss_{m}_mean'],
            'final_test_acc_mean': agg[-1][f'acc_{m}_mean'],
            'time_per_step_ms_mean': float(np.mean([r[f'time_{m}_ms'] for r in agg]))
        }
    return agg, summary

def write_csv(path, header, rows):
    "Write a CSV file given header (list) and rows (list of lists)."
    with open(path, "w") as f:
        f.write(",".join(header)+"\n")
        for r in rows: f.write(",".join(r)+"\n")

def small_lr_grid(seeds, method, lrs, epochs=20):
    "Evaluate a small learning-rate grid; return mean final accuracy over seeds for each lr."
    results = []
    for lr in lrs:
        finals=[]
        for s in seeds:
            res = run_logreg_once(seed=s, method=method, epochs=epochs, base_eta=lr)
            finals.append(res[-1][2])
        results.append((lr, float(np.mean(finals))))
    return results

def main():
    "Main entry point to generate all result files used by the manuscript."
    seeds = list(range(10))
    methods = ['SGD','SGDM','SGDCosine','RMSprop','Adam','AdamW','Align_SGD','EB_SGD']
    epochs = 20; d = 20; N_train, N_test = 2000, 1000

    agg, summary = aggregate_over_seeds_logreg(seeds, methods, epochs=epochs, d=d, N_train=N_train, N_test=N_test)

    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    meta = {
        "timestamp": ts,
        "how_to_run": "python3 simulation.py",
        "task":"Synthetic logistic regression",
        "N_train": int(N_train), "N_test": int(N_test), "d": int(d),
        "epochs": int(epochs), "seeds": seeds,
        "software": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()},
        "params": {
            "SGD":{"eta":0.05},
            "SGDM":{"eta":0.05,"momentum":0.9},
            "SGDCosine":{"eta0":0.05, "schedule":"cosine over epochs"},
            "RMSprop":{"eta":0.01,"rho":0.99,"eps":1e-8},
            "Adam":{"eta":0.02,"beta1":0.9,"beta2":0.999,"eps":1e-8},
            "AdamW":{"eta":0.02,"beta1":0.9,"beta2":0.999,"eps":1e-8,"weight_decay":1e-4},
            "Align_SGD":{"eta_base":0.05,"target_align":0.70,"beta":2.0,"W":50},
            "EB_SGD":{"eta_base":0.03,"entropy_target":0.65,"alpha":2.0,"W":50}
        }
    }
    with open("results.txt","w") as f:
        f.write("# Meta: "+json.dumps(meta)+"\n")
        header_cols = ["epoch"]
        for m in methods:
            header_cols += [f"loss_{m}_mean", f"loss_{m}_ci", f"acc_{m}_mean", f"acc_{m}_ci"]
        header_cols += ["eb_eta_mean","eb_ent_mean","align_metric_mean"]
        for m in methods:
            header_cols += [f"time_{m}_ms"]
        f.write(",".join(header_cols)+"\n")
        for r in agg:
            vals = [str(r["epoch"])]
            for m in methods:
                vals += [f'{r[f"loss_{m}_mean"]:.4f}', f'{r[f"loss_{m}_ci"]:.4f}', f'{r[f"acc_{m}_mean"]:.4f}', f'{r[f"acc_{m}_ci"]:.4f}']
            vals += [f'{r.get("eb_eta_mean", float("nan")):.4f}', f'{r.get("eb_ent_mean", float("nan")):.4f}', f'{r.get("align_metric_mean", float("nan")):.4f}']
            for m in methods:
                vals += [f'{r[f"time_{m}_ms"]:.4f}']
            f.write(",".join(vals)+"\n")

    header = "method,final_train_loss_mean,final_test_acc_mean,time_per_step_ms_mean".split(",")
    rows = []
    for m in methods:
        rows.append([m, f'{summary[m]["final_train_loss_mean"]:.4f}', f'{summary[m]["final_test_acc_mean"]:.4f}', f'{summary[m]["time_per_step_ms_mean"]:.4f}'])
    write_csv("results_summary.csv", header, rows)

    # MLP nonconvex test (subset of methods)
    mlp_methods = ['SGD','Adam','EB_SGD']; mlp_epochs = 15
    base_etas = {'SGD':0.05,'Adam':0.005,'EB_SGD':0.01}
    per_method = {m: [] for m in mlp_methods}
    for m in mlp_methods:
        for s in seeds:
            res = run_mlp_once(seed=s, method=m, epochs=mlp_epochs, base_eta=base_etas[m])
            per_method[m].append(res)
    header = ["epoch"]
    for m in mlp_methods:
        header += [f"loss_{m}_mean", f"loss_{m}_ci", f"acc_{m}_mean", f"acc_{m}_ci"]
    header += ["eb_eta_mean","eb_ent_mean"]
    header += [f"time_{m}_ms" for m in mlp_methods]
    rows=[]
    for ep in range(1, mlp_epochs+1):
        row = [str(ep)]
        for m in mlp_methods:
            losses = [per_method[m][i][ep-1][1] for i in range(len(seeds))]
            accs   = [per_method[m][i][ep-1][2] for i in range(len(seeds))]
            tms    = [per_method[m][i][ep-1][5] for i in range(len(seeds))]
            mean_l, std_l = float(np.mean(losses)), float(np.std(losses, ddof=1))
            mean_a, std_a = float(np.mean(accs)), float(np.std(accs, ddof=1))
            n=len(seeds)
            ci_l = 1.96*std_l/np.sqrt(n) if n>1 else 0.0
            ci_a = 1.96*std_a/np.sqrt(n) if n>1 else 0.0
            row += [f"{mean_l:.4f}", f"{ci_l:.4f}", f"{mean_a:.4f}", f"{ci_a:.4f}"]
        etas = [per_method['EB_SGD'][i][ep-1][3] for i in range(len(seeds))]
        ents = [per_method['EB_SGD'][i][ep-1][4] for i in range(len(seeds))]
        row += [f"{float(np.mean(etas)):.4f}", f"{float(np.mean(ents)):.4f}"]
        for m in mlp_methods:
            tms = [per_method[m][i][ep-1][5] for i in range(len(seeds))]
            row += [f"{float(1000*np.mean(tms)):.4f}"]
        rows.append(row)
    write_csv("results_mlp.txt", header, rows)

    # Ablations (sweeps)
    alphas = [0.5, 1.0, 1.5, 2.0, 3.0]
    Wvals  = [20, 50, 80, 120, 160]
    rows_alpha=[]
    for a in alphas:
        agg_a, _ = aggregate_over_seeds_logreg(seeds, ['EB_SGD'], epochs=epochs, d=d, N_train=N_train, N_test=N_test,
                                               eb_target=0.65, eb_alpha=a, W=50)
        final_acc_synth = agg_a[-1]['acc_EB_SGD_mean']
        mlp_accs=[]
        for s in seeds:
            res = run_mlp_once(seed=s, method='EB_SGD', epochs=15, base_eta=0.01, eb_target=0.60, eb_alpha=a, W=50)
            mlp_accs.append(res[-1][2])
        final_acc_mlp = float(np.mean(mlp_accs))
        rows_alpha.append([f"{a}", f"{final_acc_synth:.4f}", f"{final_acc_mlp:.4f}"])
    write_csv("ablation_alpha.csv", ["alpha","final_acc_synth","final_acc_mlp"], rows_alpha)

    rows_W=[]
    for W in Wvals:
        agg_w, _ = aggregate_over_seeds_logreg(seeds, ['EB_SGD'], epochs=epochs, d=d, N_train=N_train, N_test=N_test,
                                               eb_target=0.65, eb_alpha=2.0, W=W)
        final_acc_synth = agg_w[-1]['acc_EB_SGD_mean']
        mlp_accs=[]
        for s in seeds:
            res = run_mlp_once(seed=s, method='EB_SGD', epochs=15, base_eta=0.01, eb_target=0.60, eb_alpha=1.5, W=W)
            mlp_accs.append(res[-1][2])
        final_acc_mlp = float(np.mean(mlp_accs))
        rows_W.append([f"{W}", f"{final_acc_synth:.4f}", f"{final_acc_mlp:.4f}"])
    write_csv("ablation_W.csv", ["W","final_acc_synth","final_acc_mlp"], rows_W)

    # Target entropy sweep
    Hstars = [0.50, 0.60, 0.65, 0.70, 0.80]
    rows_H=[]
    for Hs in Hstars:
        agg_h, _ = aggregate_over_seeds_logreg(seeds, ['EB_SGD'], epochs=epochs, d=d, N_train=N_train, N_test=N_test,
                                               eb_target=Hs, eb_alpha=2.0, W=50)
        final_acc_synth = agg_h[-1]['acc_EB_SGD_mean']
        rows_H.append([f"{Hs}", f"{final_acc_synth:.4f}"])
    write_csv("ablation_H.csv", ["Hstar","final_acc_synth"], rows_H)

    # Lightweight learning-rate grid (fairness)
    grid_methods = ['SGD','SGDCosine','Adam','AdamW','EB_SGD']
    lr_grid = {
        'SGD':[0.02, 0.05, 0.10],
        'SGDCosine':[0.02, 0.05, 0.10],
        'Adam':[0.005, 0.01, 0.02],
        'AdamW':[0.005, 0.01, 0.02],
        'EB_SGD':[0.01, 0.02, 0.03]
    }
    rows=[]
    grid_seeds = list(range(5))  # keep runtime moderate
    best = {}
    for m in grid_methods:
        results = small_lr_grid(grid_seeds, m, lr_grid[m], epochs=15)
        best_lr, best_acc = None, -1.0
        for lr, acc in results:
            rows.append([m, f"{lr}", f"{acc:.4f}"])
            if acc > best_acc:
                best_acc = acc; best_lr = lr
        best[m] = (best_lr, best_acc)
    write_csv("grid_lr.csv", ["method","lr","final_acc_synth"], rows)

    # Best-of-grid summary for fairness plot
    rows_best = []
    for m in grid_methods:
        lr, acc = best[m]
        rows_best.append([m, f"{lr}", f"{acc:.4f}"])
    write_csv("best_grid.csv", ["method","best_lr","best_final_acc"], rows_best)

    print("Wrote results.txt, results_summary.csv, results_mlp.txt, ablation_alpha.csv, ablation_W.csv, ablation_H.csv, grid_lr.csv, best_grid.csv")

if __name__ == "__main__":
    main()
# === End block 1 ===
