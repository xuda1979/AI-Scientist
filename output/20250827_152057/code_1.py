import torch, math

def kl_gaussians(q_mu, q_logvar, p_mu, p_logvar):
    return 0.5 * torch.sum(
        p_logvar - q_logvar +
        (torch.exp(q_logvar) + (q_mu - p_mu)**2) / torch.exp(p_logvar) - 1.0
    )

def kl_beta(q_alpha, q_beta, p_alpha, p_beta):
    import torch.special as sp
    t1 = sp.betaln(p_alpha, p_beta) - sp.betaln(q_alpha, q_beta)
    t2 = (q_alpha - p_alpha) * (sp.digamma(q_alpha) - sp.digamma(q_alpha + q_beta))
    t3 = (q_beta  - p_beta)  * (sp.digamma(q_beta)  - sp.digamma(q_alpha + q_beta))
    return t1 + t2 + t3

def pacbayes_lambda(emp_loss, kl, n, delta, lam):
    return (emp_loss + (kl + math.log(2*math.sqrt(n)/delta)) / (lam * n)) / (1 - lam/2)

def binary_kl(a, b, eps=1e-12):
    a = min(max(a, eps), 1 - eps)
    b = min(max(b, eps), 1 - eps)
    return a*math.log(a/b) + (1-a)*math.log((1-a)/(1-b))

def kl_inverse(emp_loss, c_over_n, iters=60):
    lo, hi = emp_loss, 1.0
    for _ in range(iters):
        mid = 0.5*(lo + hi)
        if binary_kl(emp_loss, mid) <= c_over_n: hi = mid
        else: lo = mid
    return hi

def final_certificate(emp_loss, kl, n, delta):
    c_over_n = (kl + math.log((n + 1)/delta)) / n
    return kl_inverse(emp_loss, c_over_n)

def exp_opnorm_bound(vmax, n, R=2.0):
    # E||Δ||_op <= sqrt(2 v ln(2n)) + (2/3) R ln(2n)
    return math.sqrt(2.0 * vmax * math.log(2*n)) + (2.0/3.0) * R * math.log(2*n)

def empirical_bernstein_upper(mean_vals, U, delta):
    # mean_vals: tensor of sample values in [0, U]
    M = mean_vals.numel()
    mu_hat = mean_vals.mean().item()
    var_hat = mean_vals.var(unbiased=True).item() if M > 1 else 0.0
    bonus = math.sqrt(2 * var_hat * math.log(3/delta) / M) + 3*U*math.log(3/delta)/M
    return mu_hat + bonus