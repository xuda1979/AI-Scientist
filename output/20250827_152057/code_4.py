def gibbs_to_mixture_gap_krr(L, B, lam, upper_disp):
    # Gamma_KRR = L * B / lam
    return (L * B / lam) * upper_disp

def gibbs_to_mixture_gap_svm(Lg, kappa, mu, n, upper_disp):
    # Gamma_SVM = L_gamma * kappa^2 / (mu * sqrt(n))
    return (Lg * (kappa**2) / (mu * math.sqrt(n))) * upper_disp