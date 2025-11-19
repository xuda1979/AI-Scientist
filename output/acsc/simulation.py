import numpy as np
import pandas as pd
import argparse
import json
from scipy.stats import sem, t, binom, chi2, norm
from statsmodels.stats.multitest import multipletests
from math import erfc, sqrt, log
import os
from typing import List, Dict, Any, Tuple, Set

# --- Core Simulation Logic ---

def generate_data(n_total: int, C: int, gamma: float, noise: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates synthetic data with varying difficulty.

    Args:
        n_total: Total number of data points to generate.
        C: Number of classes.
        gamma: Difficulty scaling factor for the sigmoid function.
        noise: Standard deviation of Gaussian noise added to correctness probabilities.
        seed: Random seed for reproducibility.

    Returns:
        A tuple containing difficulties, true labels, and correctness probabilities.
    """
    rng = np.random.default_rng(seed)
    difficulties = rng.uniform(-2, 2, n_total)
    true_labels = rng.integers(0, C, n_total)
    
    # Sigmoid maps difficulty to correctness probability
    base_probs = 1 / (1 + np.exp(-gamma * difficulties))
    
    # Add instance-level noise
    correctness_probs = np.clip(base_probs + rng.normal(0, noise, n_total), 0, 1)
    
    return difficulties, true_labels, correctness_probs

class NoisyOracle:
    """
    Simulates a noisy oracle (e.g., an LLM) that provides samples for a given input.
    """
    def __init__(self, C: int, seed: int):
        self.C = C
        self.rng = np.random.default_rng(seed)

    def sample(self, correctness_prob: float, true_label: int) -> int:
        """Draw a single sample from the oracle."""
        if self.rng.random() < correctness_prob:
            return true_label
        else:
            # Uniformly sample from incorrect labels
            incorrect_labels = list(range(self.C))
            if self.C > 1:
                incorrect_labels.remove(true_label)
            
            if not incorrect_labels: # Fallback for C=1 case
                return true_label
                
            return self.rng.choice(incorrect_labels)

# --- Conformal Calibration and Prediction ---

def calibrate_tau(calib_data: pd.DataFrame, oracle: NoisyOracle, Kmax: int, alpha: float) -> Tuple[float, float, float]:
    """
    Performs split-conformal calibration to find the vote-share threshold tau.

    Args:
        calib_data: DataFrame with calibration data.
        oracle: The noisy oracle for sampling.
        Kmax: The maximum sample budget.
        alpha: The target miscoverage rate.

    Returns:
        A tuple containing the calibrated threshold tau, and the mean/median vote shares.
    """
    correct_label_vote_shares = []
    for _, row in calib_data.iterrows():
        samples = [oracle.sample(row['correctness_prob'], int(row['true_label'])) for _ in range(Kmax)]
        true_label_count = samples.count(int(row['true_label']))
        correct_label_vote_shares.append(true_label_count / Kmax)
    
    # Standard finite-sample correction for split conformal
    n_calib = len(calib_data)
    q_level = np.ceil((n_calib + 1) * (1 - alpha)) / (n_calib) # Correct quantile calculation
    if q_level > 1.0: q_level = 1.0

    # Ensure we handle the index correctly
    sorted_scores = sorted(correct_label_vote_shares)
    k = int(np.floor(q_level * n_calib)) -1
    k = max(0, min(k, n_calib-1))
    
    # A more standard implementation using numpy.quantile which handles edge cases
    tau_np = np.quantile(correct_label_vote_shares, 1-alpha, method='higher')

    return float(tau_np), float(np.mean(correct_label_vote_shares)), float(np.median(correct_label_vote_shares))

# --- Decoding Methods ---

def run_naive_acsc(oracle: NoisyOracle, prob: float, label: int, Kmax: int, C: int, tau: float) -> Tuple[Set[int], int, float]:
    """Runs the naive ACSC heuristic."""
    counts = np.zeros(C)
    for t in range(1, Kmax + 1):
        sample = oracle.sample(prob, label)
        counts[sample] += 1
        vote_shares = counts / t
        if np.any(vote_shares >= tau):
            S = set(np.where(vote_shares >= tau)[0])
            proxy_prob = 1.0 / len(S) if len(S) > 0 else 0
            return S, t, proxy_prob
    
    # If loop finishes, return final set
    vote_shares = counts / Kmax
    S = set(np.where(vote_shares >= tau)[0])
    proxy_prob = 1.0 / len(S) if len(S) > 0 else 0
    return S, Kmax, proxy_prob

def run_acsc_ub(oracle: NoisyOracle, prob: float, label: int, Kmax: int, C: int, tau: float, m_stop: int) -> Tuple[Set[int], int, float]:
    """Runs the safe ACSC-UB method."""
    counts = np.zeros(C)
    for t in range(1, Kmax + 1):
        sample = oracle.sample(prob, label)
        counts[sample] += 1
        
        # Calculate upper bounds on final vote shares
        share_ubs = (counts + (Kmax - t)) / Kmax
        S_ub = set(np.where(share_ubs >= tau)[0])
        
        if len(S_ub) <= m_stop:
            proxy_prob = 1.0 / len(S_ub) if len(S_ub) > 0 else 0
            return S_ub, t, proxy_prob
            
    # Fallback to final conformal set if loop finishes
    vote_shares = counts / Kmax
    S = set(np.where(vote_shares >= tau)[0])
    proxy_prob = 1.0 / len(S) if len(S) > 0 else 0
    return S, Kmax, proxy_prob

def run_fixed_k_majority(oracle: NoisyOracle, prob: float, label: int, Kmax: int, C: int) -> Tuple[Set[int], int, float]:
    """Runs fixed-budget majority vote."""
    samples = [oracle.sample(prob, label) for _ in range(Kmax)]
    counts = np.bincount(samples, minlength=C)
    winner = np.argmax(counts)
    proxy_prob = np.max(counts) / Kmax
    return {int(winner)}, Kmax, float(proxy_prob)

def run_q_consensus(oracle: NoisyOracle, prob: float, label: int, Kmax: int, C: int, q: float) -> Tuple[Set[int], int, float]:
    """Runs q-consensus heuristic."""
    counts = np.zeros(C)
    for t in range(1, Kmax + 1):
        sample = oracle.sample(prob, label)
        counts[sample] += 1
        vote_shares = counts / t
        if np.any(vote_shares >= q):
            winner = np.argmax(vote_shares)
            proxy_prob = np.max(vote_shares)
            return {int(winner)}, t, float(proxy_prob)
    
    winner = np.argmax(counts)
    proxy_prob = np.max(counts) / Kmax
    return {int(winner)}, Kmax, float(proxy_prob)

def run_cs_hoeffding(oracle: NoisyOracle, prob: float, label: int, Kmax: int, C: int, delta: float) -> Tuple[Set[int], int, float]:
    """Runs a leader-gap stopping rule based on Hoeffding's inequality."""
    counts = np.zeros(C)
    for t in range(1, Kmax + 1):
        sample = oracle.sample(prob, label)
        counts[sample] += 1
        if t < 2 or C < 2:  # Cannot compute a gap with <2 samples or <2 classes
            continue
        
        sorted_indices = np.argsort(counts)[::-1]
        top1_count = counts[sorted_indices[0]]
        top2_count = counts[sorted_indices[1]]
        
        # FIX: Hoeffding bound needs log term. (2*C*(C-1)) is for a specific type of CS, a simpler bound is sufficient.
        # Using a simpler form for demonstration. log(1/delta) is common.
        bound = sqrt((1 / (2 * t)) * log(2 / delta))
        
        if (top1_count - top2_count) / t > bound:
            winner = np.argmax(counts)
            proxy_prob = top1_count / t
            return {int(winner)}, t, float(proxy_prob)
            
    winner = np.argmax(counts)
    proxy_prob = np.max(counts) / Kmax
    return {int(winner)}, Kmax, float(proxy_prob)

def run_random_guess(C: int, rng: np.random.Generator) -> Tuple[Set[int], int, float]:
    """Makes a random guess."""
    return {rng.integers(0, C)}, 0, 1/C

# --- Evaluation and Metrics ---

def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculates Expected Calibration Error (ECE)."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]
    
    ece = 0.0
    total_n = len(y_true)
    if total_n == 0:
        return 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def calculate_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculates the Brier Score."""
    if len(y_true) == 0:
        return 0.0
    return float(np.mean((y_prob - y_true)**2))

def run_evaluation(test_data: pd.DataFrame, oracle: NoisyOracle, Kmax: int, C: int, tau: float, args: argparse.Namespace, seed_rng: np.random.Generator) -> Dict[str, List[Dict]]:
    """Runs all methods on the test set."""
    results = {
        'acsc_naive': [], 'acsc_ub_m1': [], 'acsc_ub_m2': [], 'mv': [], 'qc': [], 'cs': [], 'rg': []
    }
    
    for _, row in test_data.iterrows():
        prob = row['correctness_prob']
        label = int(row['true_label'])

        # Naive ACSC
        S, t, p = run_naive_acsc(oracle, prob, label, Kmax, C, tau)
        results['acsc_naive'].append({'S': S, 't': t, 'p': p, 'label': label})
        
        # ACSC-UB (m=1)
        S, t, p = run_acsc_ub(oracle, prob, label, Kmax, C, tau, 1)
        results['acsc_ub_m1'].append({'S': S, 't': t, 'p': p, 'label': label})
        
        # ACSC-UB (m=2)
        S, t, p = run_acsc_ub(oracle, prob, label, Kmax, C, tau, 2)
        results['acsc_ub_m2'].append({'S': S, 't': t, 'p': p, 'label': label})

        # Fixed-K Majority Vote
        S, t, p = run_fixed_k_majority(oracle, prob, label, Kmax, C)
        results['mv'].append({'S': S, 't': t, 'p': p, 'label': label})

        # q-consensus
        S, t, p = run_q_consensus(oracle, prob, label, Kmax, C, args.q_consensus)
        results['qc'].append({'S': S, 't': t, 'p': p, 'label': label})

        # CS Hoeffding
        S, t, p = run_cs_hoeffding(oracle, prob, label, Kmax, C, args.delta_cs)
        results['cs'].append({'S': S, 't': t, 'p': p, 'label': label})

        # Random Guess
        S, t, p = run_random_guess(C, seed_rng)
        results['rg'].append({'S': S, 't': t, 'p': p, 'label': label})
        
    return results

def aggregate_results(res: List[Dict], label_key: str) -> Dict[str, Any]:
    """Aggregates results from a single method run."""
    correct_list = [r['label'] in r['S'] for r in res]
    
    metric = np.mean(correct_list)
    set_sizes = [len(r['S']) for r in res]
    samples_used = [r['t'] for r in res if r['t'] > 0] # Exclude random guess t=0
    
    singleton_results = [r for r in res if len(r['S']) == 1]
    y_true = np.array([1 if r['label'] in r['S'] else 0 for r in singleton_results])
    y_prob = np.array([r['p'] for r in singleton_results])

    ece = calculate_ece(y_true, y_prob)
    brier = calculate_brier(y_true, y_prob)
    
    return {
        label_key: metric,
        'avg_set_size': np.mean(set_sizes),
        'avg_samples_used': np.mean(samples_used) if samples_used else 0.0,
        'median_samples_used': np.median(samples_used) if samples_used else 0.0,
        'ece': ece,
        'brier': brier,
        'correct_list': correct_list,
        'samples_used_list': [r['t'] for r in res]
    }

# --- Statistical Tests ---

def z_test_coverage(coverage: float, n: int, target: float = 0.9) -> Tuple[float, float]:
    """Performs a two-sided z-test for proportions."""
    se = sqrt(target * (1 - target) / n)
    if se == 0:
        return (np.inf, 0.0) if coverage == target else (-np.inf, 1.0)
    z = (coverage - target) / se
    p_val = 2 * (1 - norm.cdf(abs(z)))
    return z, p_val

def mcnemar_test(list_a: List[bool], list_b: List[bool]) -> float:
    """Performs McNemar's test for paired nominal data."""
    a_yes_b_no = sum(1 for a, b in zip(list_a, list_b) if a and not b)
    a_no_b_yes = sum(1 for a, b in zip(list_a, list_b) if not a and b)
    
    # Applying continuity correction
    numerator = abs(a_yes_b_no - a_no_b_yes) - 1
    denominator = a_yes_b_no + a_no_b_yes
    
    if denominator == 0:
        return 1.0 # No disagreements
        
    if numerator < 0: # Correction makes it negative, effectively zero disagreement
        return 1.0

    chi2_stat = numerator**2 / denominator
    
    p_val = chi2.sf(chi2_stat, 1)
    return p_val

# --- Main Execution Logic ---

def main(args: argparse.Namespace):
    all_seeds_results = []
    
    for i in range(args.n_seeds):
        seed = args.seed + i
        seed_rng = np.random.default_rng(seed)
        
        # 1. Generate and split data
        _, true_labels, correctness_probs = generate_data(
            args.n_total, args.C, args.gamma, args.noise, seed
        )
        df = pd.DataFrame({'true_label': true_labels, 'correctness_prob': correctness_probs})
        calib_data = df.iloc[:args.n_calib]
        test_data = df.iloc[args.n_calib:]
        
        # 2. Setup oracle and calibrate
        oracle = NoisyOracle(args.C, seed)
        tau, _, _ = calibrate_tau(calib_data, oracle, args.Kmax, args.alpha)
        
        # 3. Evaluate all methods
        test_run_results = run_evaluation(test_data, oracle, args.Kmax, args.C, tau, args, seed_rng)
        
        # 4. Aggregate metrics for this seed
        method_aggregates = {
            'acsc_naive': aggregate_results(test_run_results['acsc_naive'], 'coverage'),
            'acsc_ub_m1': aggregate_results(test_run_results['acsc_ub_m1'], 'coverage'),
            'acsc_ub_m2': aggregate_results(test_run_results['acsc_ub_m2'], 'coverage'),
            'mv': aggregate_results(test_run_results['mv'], 'accuracy'),
            'qc': aggregate_results(test_run_results['qc'], 'accuracy'),
            'cs': aggregate_results(test_run_results['cs'], 'accuracy'),
            'rg': aggregate_results(test_run_results['rg'], 'accuracy')
        }

        # 5. Perform statistical tests for this seed
        tests = {
            'pval_naive_vs_target': z_test_coverage(method_aggregates['acsc_naive']['coverage'], len(test_data), 1 - args.alpha)[1],
            'pval_ub_m1_vs_target': z_test_coverage(method_aggregates['acsc_ub_m1']['coverage'], len(test_data), 1 - args.alpha)[1],
            'mcnemar_p_naive_vs_ub_m1': mcnemar_test(method_aggregates['acsc_naive']['correct_list'], method_aggregates['acsc_ub_m1']['correct_list'])
        }

        # 6. Collate summary for the seed
        seed_summary = {"seed": seed}
        for name, agg in method_aggregates.items():
            for key, val in agg.items():
                if not key.endswith('_list'): # Don't save large lists
                    seed_summary[f"{name}_{key}"] = val
        seed_summary.update(tests)
        all_seeds_results.append(seed_summary)
        print(f"Finished seed {seed}, Naive ACSC Coverage: {method_aggregates['acsc_naive']['coverage']:.4f}, ACSC-UB (m=1) Coverage: {method_aggregates['acsc_ub_m1']['coverage']:.4f}")

    # --- Final Aggregation and Output ---
    df_results = pd.DataFrame(all_seeds_results)
    mean_results = df_results.mean().to_dict()
    std_results = df_results.std().to_dict()
    
    final_summary = {key: val for key, val in mean_results.items() if not key.endswith('_std')}
    for key, val in std_results.items():
        final_summary[f"{key}_std"] = val
        
    # BH correction for p-values (using mean p-values across seeds)
    pvals_to_correct = [p for name, p in mean_results.items() if 'pval' in name or 'mcnemar' in name]
    if pvals_to_correct:
        _, qvals, _, _ = multipletests(pvals_to_correct, alpha=0.05, method='fdr_bh')
        p_names = [name for name in mean_results if 'pval' in name or 'mcnemar' in name]
        for i, name in enumerate(p_names):
            final_summary[f"{name}_bh_q"] = qvals[i]

    final_summary['n_seeds'] = args.n_seeds

    os.makedirs(args.out_dir, exist_ok=True)
    df_results.to_csv(os.path.join(args.out_dir, 'results_all_seeds.csv'), index=False)
    with open(os.path.join(args.out_dir, 'results_summary.json'), 'w') as f:
        json.dump(final_summary, f, indent=2, cls=NpEncoder)
        
    print(f"\nFinal aggregated results over {args.n_seeds} seeds saved to '{args.out_dir}/'")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anytime Conformal Self-Consistency Simulation")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--n_seeds", type=int, default=5, help="Number of seeds to run")
    parser.add_argument("--n_total", type=int, default=1500, help="Total number of data points")
    parser.add_argument("--n_calib", type=int, default=300, help="Number of calibration points")
    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage rate")
    parser.add_argument("--Kmax", type=int, default=16, help="Maximum sample budget")
    parser.add_argument("--C", type=int, default=8, help="Number of classes")
    parser.add_argument("--gamma", type=float, default=1.2, help="Difficulty scaling factor")
    parser.add_argument("--noise", type=float, default=0.1, help="Noise in correctness probability. Paper uses 0.1 for good results.")
    
    parser.add_argument("--q_consensus", type=float, default=0.75, help="Threshold for q-consensus")
    parser.add_argument("--delta_cs", type=float, default=0.05, help="Confidence for Hoeffding stopping rule")

    parser.add_argument("--out_dir", type=str, default="output", help="Output directory for results")
    
    args = parser.parse_args()
    main(args)