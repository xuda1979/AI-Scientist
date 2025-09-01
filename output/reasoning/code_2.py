def tokenizer_self_test(tok, manifest_db, protected_bytes, norms=("NFC","NFD","NFKC","NFKD")):
    key = (tok.name, tok.version, tok.flags())
    if key in manifest_db: 
        return manifest_db[key]
    verdict = run_detector_suite(tok, protected_bytes, norms=norms, horizons=(3,4,5))
    manifest_db[key] = verdict
    if verdict["status"] != "safe":
        policy = {"elevate_horizon": verdict.get("min_safe_L", 4),
                  "guard_spans": True,
                  "disable_sampling": True,
                  "byte_fallback_on_ambiguous": True}
    else:
        policy = {"elevate_horizon": 3, "guard_spans": False}
    manifest_db[key]["policy"] = policy
    return manifest_db[key]