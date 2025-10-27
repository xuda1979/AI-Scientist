1. Manuscript Outline [CRITICAL]
   1. Introduction: motivation, research question, contributions.
   2. Related Literature: digital adoption/productivity; asset pricing and profitability; China AI policy context.
   3. Data: firm universe, sources (patents, job postings, supplier disclosures), financials, returns, policy shocks, controls.
   4. AI Adoption Measurement: construction, NLP/classification, validation.
   5. Identification Strategy: DiD design around 2017 plan and 2023 GenAI measures; event study.
   6. Productivity & Profitability Effects: TFP, ROIC, margins.
   7. Asset-Pricing Tests: AI adoption factor, cross-sectional pricing in A-shares and STAR.
   8. Robustness & Mechanisms: placebo tests, instruments, heterogeneity.
   9. Prior Art Differentiation.
   10. Reproducibility.
   11. Conclusion.
   12. Appendices: variable definitions, codebook, additional tests.

2. Core Thesis & Contributions [CRITICAL]
   - Thesis: Firm-level AI adoption in China, measured via patents, job postings, and supplier ties, causally raises TFP and profitability and commands a priced risk/characteristic in A-shares and STAR.
   - Contributions:
     1) Construct a firm-level AI adoption index for Chinese listed firms from multiple sources with validated NLP.
     2) Causal estimates of AI’s impact on TFP, ROIC, and margins using DiD around the 2017 Next Generation AI Plan and 2023 Interim Measures on Generative AI.
     3) Asset-pricing link: an “AI adoption” factor and characteristic that explain cross-sectional expected returns beyond standard China factors.

3. Data & Measurement [CRITICAL]
   - Firms/financials/returns: CSMAR, WIND; A-shares and STAR Market; 2009–2024.
   - Patents: CNIPA (and Derwent if available). Classify AI using OECD AI taxonomy keywords and AI Index taxonomy.
   - Job postings: Zhaopin, 51job, Liepin, Boss Zhipin; parse titles/descriptions for AI skills, cloud/ML tooling.
   - Supplier disclosures: annual reports (Shanghai/Shenzhen exchanges), extract mentions of AI/cloud vendors (e.g., Alibaba Cloud, Baidu, Huawei Cloud).
   - AI Adoption Index: weighted composite of standardized patent intensity, AI hiring intensity, supplier AI exposure. Validate via manual coding (random 500 firm-year sample), precision/recall targets ≥0.85.

4. Empirical Strategy [CRITICAL]
   - TFP: estimate via Olley-Pakes, Levinsohn-Petrin, and Ackerberg-Caves-Frazer; Wooldridge proxy as benchmark.
   - DiD: staggered adoption; event-study with Sun–Abraham and Callaway–Sant’Anna estimators; firm and firm×industry×trend FE; exposure instrumented by pre-policy AI readiness (provincial data centers/5G rollout).
   - Outcomes: TFP, ROIC, operating margin, asset turnover; mechanisms via SG&A IT share.
   - Asset pricing: construct AI factor (H–L on AI index within industry-size buckets), test alphas vs China Fama–French 5, Hou–Xue–Zhang q-factors; Fama–MacBeth with controls (size, book-to-market, investment, expected profitability).

5. Quantitative Evaluation [IMPORTANT]
   - Metrics: DiD coefficients, dynamic ATT, R^2; portfolio alphas (bps/month), Sharpe, Information Ratio; cross-sectional slopes.
   - Baselines: digital adoption proxies (IT capital intensity), general patent intensity, R&D/sales.
   - Ablations: single-source AI measures; alternative AI dictionaries; excluding Big Tech suppliers; reweightings.
   - Tests: Newey–West, cluster 2-way (firm×year-week), randomization inference; GRS, spanning, Hansen–Jagannathan distance; pre-trend F-tests.

6. Figures/Tables [IMPORTANT]
   - Fig: Policy timeline—identification windows.
   - Fig: AI index construction flow—measurement transparency.
   - Fig: Heatmap by province/industry—heterogeneity.
   - Fig: Event-study plots—parallel trends and dynamics.
   - Fig: Factor cumulative returns—risk premia.
   - Tab: Measure validation (precision/recall)—measurement quality.
   - Tab: DiD main effects—economic magnitudes.
   - Tab: Portfolio alphas and GRS—pricing power.
   - Tab: Double-sorts (AI×size/profitability)—orthogonality.

7. Prior Art Differentiation [CRITICAL]
   - Resurrecting the Size Effect (2009): We isolate AI adoption from size by within-industry-size portfolio sorts and controls; show AI factor alphas persist after size and profitability shocks; placebo using size-neutralized AI.
   - Digital Adoption→Productivity (OECD figure): Moves from aggregate correlations to firm-level causal DiD tied to China’s AI policies; richer micro measures and event dynamics.
   - Expected Profitability and Returns (2019): Distinguish AI-specific adoption from generic profitability; show incremental explanatory power and spanning tests where AI factor is not spanned by expected profitability factors.

8. Risks, Assumptions, Open Questions [IMPORTANT]
   - Endogeneity of adoption; measurement error in postings; policy anticipation.
   - Assume policy shocks shift marginal AI costs broadly; verify no confounding contemporaneous reforms.
   - Open: Channels (cloud vs in-house), SOE vs private heterogeneity.

9. Experimental Innovation Hooks [NICE-TO-HAVE]
   - Earnings-call text to validate adoption timing.
   - Exogenous compute shocks (regional data center outages) as instruments.
   - STAR vs ChiNext contrast; export exposure heterogeneity.
   - Market reaction to firm-specific AI announcements (event studies).

10. Reproducibility [CRITICAL]
   - Pre-register estimands and model specs; release code/labels; fixed random seeds; versioned dictionaries; detailed variable codebook; replication scripts for each table/figure.

11. Feasibility & Risk Mitigation [CRITICAL]
   - Resources: CSMAR/WIND, CNIPA bulk; scrape postings with stored snapshots; GPU for NLP.
   - Fallbacks: If postings sparse, rely on patents+suppliers; if policy timing fuzzy, use continuous exposure and instrumental variation; if STAR data thin, broaden to ChiNext.

Top-5 Drafting Checklist
1) Precisely define and validate the AI adoption index (sources, NLP, manual audit).
2) Present DiD with Sun–Abraham and Callaway–Sant’Anna, clear event-study pre-trends.
3) Report TFP/ROIC/margin effects with multiple production function estimators.
4) Build and test an AI factor and characteristic against China Fama–French and q-factors; include GRS/spanning.
5) Explicitly differentiate from size, generic digital adoption, and expected profitability literatures.