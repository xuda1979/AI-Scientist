#!/usr/bin/env python3
import argparse, sys, subprocess, shlex, json, hashlib, os
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(
        description="Wrapper to regenerate all ASCII tables used by the HMC manuscript.")
    ap.add_argument("--s-initial", type=int, default=12, help="Initial SBH entropy (proxy units).")
    ap.add_argument("--steps", type=int, default=12, help="Number of discrete emission steps.")
    ap.add_argument("--num-runs", type=int, default=100, help="Ensemble size for page curve.")
    ap.add_argument("--g2-runs", type=int, default=200, help="Bootstrap runs for g2 confidence intervals.")
    ap.add_argument("--kfolds", type=int, default=5, help="K folds for CV summary.")
    ap.add_argument("--tau-mem", type=float, default=8.0, help="Memory time constant for g2 model.")
    ap.add_argument("--l-mem", type=int, default=6, help="Memory depth proxy for ablations/PT-MPO.")
    ap.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    ap.add_argument("--threads", type=int, default=1, help="Thread cap for BLAS/numexpr backends.")
    ap.add_argument("--pythonhashseed", type=int, default=0, help="PYTHONHASHSEED for hash-stable dicts.")
    # Two names for the same option, to match the manuscript text
    ap.add_argument("--save-ledger", type=str, default="seed_ledger.json",
                    help="Path to write the seed ledger JSON (alias: --seed_ledger).")
    ap.add_argument("--seed_ledger", dest="save_ledger", type=str,
                    help="Alias of --save-ledger for consistency with the paper.")
    ap.add_argument("--outdir", type=str, default=".",
                    help="Output directory for tables and manifests.")
    ap.add_argument("--floatfmt", type=str, default="fixed6", choices=["fixed6","g"],
                    help="Numeric formatting for table entries (forwarded to simulation.py).")
    ap.add_argument("--dump-raw", type=str, default=None, choices=["none","csv","npy","both"],
                    help="Forwarded to simulation.py: write raw matrices for key datasets.")
    ap.add_argument("--verify-checksums", action="store_true",
                    help="After generation, compute SHA256 for outputs and compare to checksums file(s).")
    ap.add_argument("--print-manifest", action="store_true",
                    help="After generation, print a compact manifest (path, size, sha256) for produced files.")
    args = ap.parse_args()

    cmd = [
        sys.executable, "simulation.py",
        "--s-initial", str(args.s_initial),
        "--steps", str(args.steps),
        "--num-runs", str(args.num_runs),
        "--g2-runs", str(args.g2_runs),
        "--kfolds", str(args.kfolds),
        "--tau-mem", str(args.tau_mem),
        "--l-mem", str(args.l_mem),
        "--seed", str(args.seed),
        "--threads", str(args.threads),
        "--pythonhashseed", str(args.pythonhashseed),
        "--save-ledger", str(args.save_ledger),
        "--outdir", str(args.outdir),
        "--floatfmt", str(args.floatfmt),
    ]
    if args.dump_raw is not None:
        cmd += ["--dump-raw", args.dump_raw]

    print(f"+ {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.check_call(cmd)
    print(f"Done: generated datasets into {args.outdir} ; seed ledger → {args.save_ledger}")

    # Post-run: optional checksum verification and manifest printing
    try:
        ledger_path = Path(args.save_ledger)
        outdir = Path(args.outdir)
        if ledger_path.exists():
            with open(ledger_path, "r", encoding="utf-8") as f:
                ledger = json.load(f)
        else:
            ledger = {}
        files = [outdir / p for p in ledger.get("files", [])]
    except Exception as e:
        files = []
        print(f"[warn] could not read ledger for manifest: {e}", file=sys.stderr)

    def sha256_of(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    if args.print_manifest and files:
        print("\n# Manifest (path size sha256):")
        for p in files:
            try:
                print(p, p.stat().st_size, sha256_of(p))
            except FileNotFoundError:
                print(p, "MISSING", "-", file=sys.stderr)

    if args.verify_checksums:
        # Compare against checksums if present
        ok = True
        checksums_txt = [outdir / "checksums.sha256.txt", outdir / "checksums_v5.txt"]
        existing = [p for p in checksums_txt if p.exists()]
        if not existing and files:
            # compute our own ad-hoc verification if no reference file exists
            print("[info] no checksums file found; computing fresh hashes for produced files.")
            for p in files:
                try:
                    _ = sha256_of(p)
                except FileNotFoundError:
                    ok = False
                    print(f"[error] missing file listed in ledger: {p}", file=sys.stderr)
        else:
            for ref in existing:
                with open(ref, "r", encoding="utf-8") as f:
                    for line in f:
                        line=line.strip()
                        if not line: continue
                        # supports formats: "<hash>  <filename>" or "<filename> <hash>"
                        parts = line.split()
                        if len(parts) < 2: continue
                        h_expected, fn = (parts[0], parts[-1]) if len(parts[0])==64 else (parts[-1], parts[0])
                        p = outdir / Path(fn).name
                        if not p.exists() or sha256_of(p) != h_expected:
                            ok = False
                            print(f"[mismatch] {p}", file=sys.stderr)
        if not ok:
            sys.exit("[verify] checksum verification failed.")

if __name__ == "__main__":
    main()
