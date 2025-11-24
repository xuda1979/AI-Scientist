import math
import sys
import time

def t_pi_logsafe(N: int) -> float:
    """
    Compute the single‑sum T–π approximation
        π ≈ log(N) / Σ_{n=1}^N binom(2n,n)^2 / 16^n
    entirely in log‑space so that it works for very large N.
    """
    log_binom = 0.0               # log binom(0,0) = 0
    log_sum   = float('-inf')     # log(0)  (we'll build log‑sum with log‑sum‑exp)
    log16     = math.log(16)

    for n in range(1, N + 1):
        # ---- update log binomial via the correct recurrence ----
        # binom(2n,n) / binom(2n-2,n-1) = 4 - 2/n
        log_binom += math.log(4.0 - 2.0 / n)

        # log of the nth summand: 2*log_binom  -  n*log16
        log_term = 2.0 * log_binom - n * log16

        # ---- log‑sum‑exp accumulation ----
        if log_sum == float('-inf'):
            log_sum = log_term
        else:
            # ensure log_sum ≥ log_term before calling log1p
            if log_term > log_sum:
                log_sum, log_term = log_term, log_sum
            log_sum = log_sum + math.log1p(math.exp(log_term - log_sum))

    # S = e^{log_sum};   π ≈ log(N) / S = log(N) * e^{‑log_sum}
    return math.log(N) * math.exp(-log_sum)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage:  python numerical_experiments.py <N>")
        sys.exit(1)

    N = int(sys.argv[1])
    print(f"Computing T–π formula with N = {N:,} (log‑safe)…")
    t0 = time.time()
    pi_approx = t_pi_logsafe(N)
    t1 = time.time()

    print(f"\nApproximation of π : {pi_approx}")
    print(f"True value of π    : {math.pi}")
    print(f"Absolute error     : {abs(pi_approx - math.pi)}")
    print(f"Elapsed time       : {t1 - t0:.2f} s")


if __name__ == "__main__":
    main()
