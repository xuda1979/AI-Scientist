"""
Simulation for: Pricing AI Adoption in China: Firm-Level Productivity, Profitability, and Expected Returns

This script generates a synthetic panel of Chinese listed firms (2009–2024), two AI policy shocks (2017, 2023),
a multi-source AI adoption index, firm outcomes (TFP, ROIC, margins), and asset-pricing tests tailored to A-share/STAR style.

It reports:
- TWFE DiD estimates and event-study dynamics for TFP, ROIC, margins
- Portfolio and factor results for an AI adoption factor, including alpha vs FF5, Sharpe, and GRS test
- Ablations for alternative AI index constructions and a placebo (pre-policy)
- An IV-style first-stage diagnostic for policy x infrastructure instruments

Outputs are printed to the console. (CSV file saving disabled for security constraints.)
"""
import numpy as np
import pandas as pd
from numpy.random import default_rng
import statsmodels.api as sm
from scipy.stats import f as fdist

def run_sim(seed=42):
    rng = default_rng(seed)

    # Panel setup
    years = np.arange(2009, 2025)
    T = len(years)
    N = 800
    industries = np.array(["Manufacturing","IT","Health","Finance","Energy","Consumer"]) 
    provinces = np.array(["Beijing","Shanghai","Guangdong","Zhejiang","Sichuan","Jiangsu"]) 
    firm_ids = np.arange(N)
    df = pd.DataFrame({
        "firm": np.repeat(firm_ids, T),
        "year": np.tile(years, N),
    })

    # Assign firm attributes
    df["industry"] = np.repeat(rng.choice(industries, size=N, replace=True), T)
    df["province"] = np.repeat(rng.choice(provinces, size=N, replace=True), T)
    df["soestate"] = np.repeat(rng.binomial(1, 0.25, size=N), T)

    # Pre-policy "AI readiness" instrument proxy based on province infra
    prov_infra = {p: b for p, b in zip(provinces, rng.uniform(0.3, 1.0, size=len(provinces)))}
    df["infra"] = df["province"].map(prov_infra)

    # Baseline firm and year heterogeneity
    firm_fe = rng.normal(0, 0.2, size=N)
    df["firm_fe"] = np.repeat(firm_fe, T)
    year_fe = {y: v for y, v in zip(years, rng.normal(0, 0.05, size=T))}
    df["year_fe"] = df["year"].map(year_fe)

    # Policy shocks
    df["post2017"] = (df["year"] >= 2017).astype(int)
    df["post2023"] = (df["year"] >= 2023).astype(int)

    # AI adoption components (patents, hiring, suppliers)
    base_adopt = rng.uniform(0, 0.2, size=N)
    df["base_adopt"] = np.repeat(base_adopt, T)

    # Heterogeneity by infra and industry
    ind_weight = df["industry"].map({
        "IT": 1.0, "Manufacturing": 0.6, "Health": 0.7, "Finance": 0.8, "Energy": 0.4, "Consumer": 0.5
    })
    df["ind_weight"] = ind_weight.values

    for comp in ["pat", "hire", "supp"]:
        level = df["base_adopt"] + 0.1*df["firm_fe"] + 0.05*df["infra"]
        shock2017 = 0.25*df["post2017"]*(0.6*df["infra"] + 0.4*df["ind_weight"])
        shock2023 = 0.15*df["post2023"]*(0.5*df["infra"] + 0.5*df["ind_weight"])
        noise = rng.normal(0, 0.05, size=len(df))
        df[comp] = np.clip(level + shock2017 + shock2023 + noise, 0, None)

    # Composite AI index (z-scored by year, then min-max to [0,1])
    def z_by_year(series: pd.Series) -> pd.Series:
        g = series.groupby(df["year"])
        return (series - g.transform("mean")) / g.transform("std")

    df["AI_index_raw"] = 0.4*z_by_year(df["pat"]) + 0.35*z_by_year(df["hire"]) + 0.25*z_by_year(df["supp"])
    g = df.groupby("year")["AI_index_raw"]
    df["AI_index"] = (df["AI_index_raw"] - g.transform("min"))/(g.transform(lambda s: s.max()-s.min()+1e-8))

    # Outcomes: TFP (log), ROIC (%), Margin (%), only activated post-policy
    ai_change = df["AI_index"] - df.groupby("firm")["AI_index"].transform("first")
    effect_post2017 = ai_change * df["post2017"]
    effect_post2023 = ai_change * df["post2023"]
    df["lnTFP"] = 0.025*effect_post2017 + 0.010*effect_post2023 + df["firm_fe"] + df["year_fe"] + rng.normal(0, 0.04, size=len(df))
    df["ROIC"] = 5 + (0.9*effect_post2017 + 0.3*effect_post2023)*100 + 1.5*df["firm_fe"] + 0.2*(df["industry"]=="IT").astype(int) + rng.normal(0, 3, size=len(df))
    df["OMargin"] = 15 + (0.8*effect_post2017 + 0.3*effect_post2023)*100 + 1.0*(df["industry"].isin(["IT","Finance"]).astype(int)) + rng.normal(0, 2.2, size=len(df))

    # Two-way FE DiD helper
    def twfe_custom(y: str, treat_series: pd.Series):
        Xfirm = pd.get_dummies(df["firm"].astype(str), drop_first=True)
        Zyear = pd.get_dummies(df["year"].astype(str), drop_first=True)
        Y = df[y].values
        D = treat_series.values.reshape(-1,1)
        Xmat = np.hstack([D, Xfirm.values, Zyear.values])
        model = sm.OLS(Y, sm.add_constant(Xmat)).fit(cov_type="cluster", cov_kwds={"groups": df["firm"]})
        # params and bse can be numpy arrays if exog is numpy; use positional indexing
        beta = float(model.params[1])
        se = float(model.bse[1])
        return beta, se, model

    b_tfp, se_tfp, m_tfp = twfe_custom("lnTFP", df["AI_index"])
    b_roic, se_roic, m_roic = twfe_custom("ROIC", df["AI_index"])
    b_mgn, se_mgn, m_mgn = twfe_custom("OMargin", df["AI_index"])

    # Event study around 2017 using continuous exposure
    df["event_time"] = df["year"] - 2017
    bins = np.arange(-5, 7)
    rows = []
    for k in bins:
        sel = df["event_time"] == k
        if sel.any():
            y = df.loc[sel, "lnTFP"].values
            X = sm.add_constant(df.loc[sel, "AI_index"].values)
            est = sm.OLS(y, X).fit()
            rows.append((int(k), float(est.params[1]), float(est.bse[1])))
        else:
            rows.append((int(k), np.nan, np.nan))
    att = pd.DataFrame(rows, columns=["k","beta","se"])

    # Asset pricing simulation
    months = pd.date_range("2010-01-31","2024-12-31", freq="ME")
    M = len(months)
    ai_year = df.pivot(index="firm", columns="year", values="AI_index")
    ai_month = ai_year.reindex(columns=np.unique(df["year"])).interpolate(axis=1).reindex(columns=years)
    ai_month = np.repeat(ai_month.values, M//T + 1, axis=1)[:, :M]
    size_char = rng.normal(0,1,size=N)

    # China FF5-like factors: realistic means and noise
    factors = rng.normal(0,1,size=(M,5))*0.01 + np.array([0.005, 0.002, 0.001, 0.0015, 0.001]).reshape(1,-1)
    true_ai_prem = 0.0035  # 35 bps/month
    ai_noise = rng.normal(0, 0.01, size=M)
    # Keep AI factor approximately orthogonal to FF5 to preserve alpha magnitude
    ai_factor = true_ai_prem + ai_noise

    betas = rng.normal(1,0.3,size=(N,5))
    ai_loading = 0.4*(size_char - size_char.mean()) + 0.6*(ai_year.mean(axis=1) - ai_year.values.mean())
    ai_loading = (ai_loading - ai_loading.mean())/ai_loading.std()
    rets = []
    for t in range(M):
        mu = (betas @ factors[t]) + ai_loading*ai_factor[t] + rng.normal(0,0.02,size=N)
        rets.append(mu)
    rets = np.array(rets)

    # AI factor alpha vs FF5
    Xts = sm.add_constant(factors)
    res = sm.OLS(ai_factor, Xts).fit(cov_type="HAC", cov_kwds={"maxlags":3})
    alpha_ai = float(res.params[0])
    alpha_ai_t = float(res.tvalues[0])
    sharpe_ai = float(ai_factor.mean()/ai_factor.std()*np.sqrt(12))

    # Fama-MacBeth slope on AI characteristic
    chars = ai_month.T
    lmbdas = []
    for t in range(M):
        y = rets[t]
        Xcs = sm.add_constant(chars[t])
        cs = sm.OLS(y, Xcs).fit()
        lmbdas.append(float(cs.params[1]))
    lmbdas = np.array(lmbdas)
    lambda_mean = float(lmbdas.mean())
    lambda_t = float(lambda_mean/(lmbdas.std()/np.sqrt(M)))

    # GRS test on 25 size x profitability portfolios
    n_sorts = 5
    sz_q = pd.qcut(size_char, n_sorts, labels=False, duplicates="drop")
    prof = rng.normal(0,1,size=N)
    pf_q = pd.qcut(prof, n_sorts, labels=False, duplicates="drop")
    portR = np.zeros((M, n_sorts*n_sorts))
    for i in range(n_sorts):
        for j in range(n_sorts):
            mask = (sz_q==i) & (pf_q==j)
            portR[:, i*n_sorts+j] = rets[:, mask].mean(axis=1)
    alphas = []
    resids = []
    for j in range(portR.shape[1]):
        rr = portR[:,j]
        fit = sm.OLS(rr, Xts).fit(cov_type="HAC", cov_kwds={"maxlags":3})
        alphas.append(float(fit.params[0]))
        resids.append(fit.resid)
    alphas = np.array(alphas)
    resids = np.column_stack(resids)
    Tn = M
    L = portR.shape[1]
    K = factors.shape[1]
    Sigma = np.cov(resids.T)
    mu_f = factors.mean(axis=0)
    Fcov = np.cov(factors.T)
    grs_num = (Tn - L - K)/L * (alphas @ np.linalg.inv(Sigma) @ alphas)
    grs_den = 1 + (mu_f @ np.linalg.inv(Fcov) @ mu_f)
    grs_stat = float(grs_num / grs_den)
    pval_grs = float(1 - fdist.cdf(grs_stat, L, Tn - L - K))

    # Ablations: alternative AI index and single-source variants
    variants = {
        "AI_base": df["AI_index"],
        "AI_alt_w": (0.6*z_by_year(df["pat"]) + 0.2*z_by_year(df["hire"]) + 0.2*z_by_year(df["supp"])),
        "AI_pat_only": z_by_year(df["pat"]),
        "AI_hire_only": z_by_year(df["hire"]),
        "AI_supp_only": z_by_year(df["supp"]),
    }
    # Scale each variant to [0,1] within year
    for k in list(variants.keys()):
        s = variants[k].copy()
        gk = s.groupby(df["year"])
        s = (s - gk.transform("min"))/(gk.transform(lambda x: x.max()-x.min()+1e-8))
        variants[k] = s

    ablation = {}
    for name, series in variants.items():
        b, se, _ = twfe_custom("lnTFP", series)
        ablation[name] = {"beta_tfp": float(b), "se": float(se)}

    # Placebo: pre-2017 only
    pre = df["year"] <= 2016
    y_pre = df.loc[pre, "lnTFP"]
    X_pre = sm.add_constant(df.loc[pre, "AI_index"])
    placebo = sm.OLS(y_pre, X_pre).fit()
    placebo_beta = float(placebo.params[1])
    placebo_se = float(placebo.bse[1])

    # IV-style first-stage: AI_index on post2017 x infra and post2023 x infra
    ivX = sm.add_constant(pd.DataFrame({
        "post2017_infra": df["post2017"]*df["infra"],
        "post2023_infra": df["post2023"]*df["infra"],
        "ind_weight": df["ind_weight"],
    }))
    first = sm.OLS(df["AI_index"], ivX).fit()
    # Joint F-stat for the two instruments
    R = np.zeros((2, len(first.params)))
    R[0,1] = 1.0  # post2017_infra
    R[1,2] = 1.0  # post2023_infra
    r = np.zeros(2)
    ftest = first.f_test((R, r))
    first_stage_F = float(ftest.fvalue)
    first_stage_p = float(ftest.pvalue)

    # Aggregate outputs
    out = {
        "seed": seed,
        "b_tfp": float(b_tfp), "se_tfp": float(se_tfp),
        "b_roic": float(b_roic), "se_roic": float(se_roic),
        "b_mgn": float(b_mgn), "se_mgn": float(se_mgn),
        "alpha_ai": float(alpha_ai), "alpha_ai_t": float(alpha_ai_t), "sharpe_ai": float(sharpe_ai),
        "lambda_mean": float(lambda_mean), "lambda_t": float(lambda_t),
        "grs_stat": float(grs_stat), "pval_grs": float(pval_grs),
        "placebo_beta": float(placebo_beta), "placebo_se": float(placebo_se),
        "first_stage_F": float(first_stage_F), "first_stage_p": float(first_stage_p),
        "att": att, "ablation": ablation
    }
    return out

if __name__ == "__main__":
    seeds = [42, 7, 123]
    summaries = []
    for s in seeds:
        res = run_sim(seed=s)
        summaries.append(res)
        print("=== Seed {} ===".format(s))
        print("DiD (TWFE): lnTFP beta {:.4f} (se {:.4f}); ROIC {:.2f} (se {:.2f}); OMargin {:.2f} (se {:.2f})".format(
            res["b_tfp"], res["se_tfp"], res["b_roic"], res["se_roic"], res["b_mgn"], res["se_mgn"]))
        print("AI factor alpha vs FF5: {:.2f} bps/mo (t={:.2f}), Sharpe={:.2f}".format(res["alpha_ai"]*100, res["alpha_ai_t"], res["sharpe_ai"]))
        print("Fama-MacBeth slope on AI characteristic: {:.2f} bps (t={:.2f})".format(res["lambda_mean"]*100, res["lambda_t"]))
        print("GRS test (FF5 only) on 25 portfolios: stat={:.2f}, p-value={:.3f}".format(res["grs_stat"], res["pval_grs"]))
        print("Placebo pre-2017: beta {:.4f} (se {:.4f})".format(res["placebo_beta"], res["placebo_se"]))
        print("First-stage (policy x infra) joint F-test: F={:.2f}, p={:.4f}".format(res["first_stage_F"], res["first_stage_p"]))
        print("Event-study sample (k=-5..-3):")
        print(res["att"].head(3).to_string(index=False))
        print("---")

    # Compact summary (CSV saving disabled per security constraints)
    df_sum = pd.DataFrame([{
        "seed": r["seed"],
        "beta_tfp": r["b_tfp"], "se_tfp": r["se_tfp"],
        "alpha_ai_bps": r["alpha_ai"]*100, "alpha_ai_t": r["alpha_ai_t"], "sharpe_ai": r["sharpe_ai"],
        "grs_stat": r["grs_stat"], "grs_p": r["pval_grs"],
        "placebo_beta": r["placebo_beta"]
    } for r in summaries])

    # Print summary instead of writing to disk
    print("\nSummary table (first 5 rows):")
    print(df_sum.head().to_string(index=False))
