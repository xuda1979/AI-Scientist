# Auto-generated from LaTeX code blocks; consolidate all simulation here.
# === Begin extracted block 1 ===
# Shadow byte-level backoff with resynchronization and caps.
def shadow_backoff(model, tok, prod, beams, max_byte_steps=8, min_resync_k=2):
    # Each beam carries (prod_state, token_ids, score, byte_budget)
    shadow = [(s, seq, score, max_byte_steps) for (s, seq, score) in beams]
    best = None
    while shadow:
        s, seq, score, bud = pop_best(shadow)
        if bud == 0: 
            continue
        # Emit one byte guarded by protected-boundary spans
        for byte in legal_bytes_under_guard(tok):
            seq_b = seq + [tok.byte_to_token(byte)] if tok.has_byte_token(byte) else seq
            s2 = prod.transition_bytes(s, byte)
            if not s2: 
                continue
            k_legal = count_legal_tokens(prod, s2)
            if k_legal >= min_resync_k:
                # Resynchronization: return promoted beams
                yield [(s2, seq_b, score + logp_byte(byte))]
            push(shadow, (s2, seq_b, score + logp_byte(byte), bud - 1))
    yield []
# === End block 1 ===

# === Begin extracted block 2 ===
# Tokenizer self-test and verdict cache with cross-lingual adversarial contexts.
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
# === End block 2 ===

# === Begin extracted block 3 ===
# Adaptive masked training with leakage target and mixture for calibration.
def adaptive_mask_training(model, data, prod_states, type_states, target_illegal=1e-4, margin=2.0, 
                           temp=1.0, lambdas=(0.0,0.25,0.5,0.75,1.0), schedule_steps=1000, seed=7):
    torch.manual_seed(seed)
    B = 30.0; alpha = 1.0; lam_idx = 2
    opt = AdamW(model.parameters(), lr=2e-4)
    for step, batch in enumerate(dataloader(data)):
        lam = lambdas[min(lam_idx, len(lambdas)-1)]
        masked = np.random.rand() < lam
        logits = model.forward(batch.inputs) / temp
        if masked:
            legal = compute_legal_sets(prod_states, type_states, batch.prefix_states)
            eps = choose_smoothing(legal, min_eps=0.02, max_eps=0.1)
            m = soft_mask_from_legal(legal, alpha, B=B)
            logits = logits + m
            loss = smoothed_cross_entropy(logits, batch.targets, eps, restrict_to_legal=legal)
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                illegal_mass = (probs * illegal_indicator(legal)).sum(dim=-1).mean().item()
                if illegal_mass > target_illegal / margin:
                    B = min(60.0, B + 2.0)   # adaptively increase B
                elif step % schedule_steps == 0 and lam_idx < len(lambdas)-1:
                    lam_idx += 1             # anneal toward more masking
                log_metrics({"illegal_mass": illegal_mass, "B": B, "lambda": lam})
        else:
            loss = cross_entropy(logits, batch.targets)
        loss.backward(); opt.step(); opt.zero_grad()
# === End block 3 ===

# === Begin extracted block 4 ===
# OIP-CAD decoding with invariant checks and SLA accounting.
def oip_cad_decode(model, tok, product, types, prompt, oip=None, beam=4, sla_ms=80, det=True):
    verdict = tokenizer_self_test(tok, MANIFEST_DB, protected_bytes=GUARDED)
    policy = verdict["policy"]
    t0 = now_ms(); stats = {"forward":0.0, "mask":0.0, "backoff":0.0, "repair":0.0}
    beams = [(product.start(), tok.encode(prompt), 0.0)]
    state = "Decode"; shadow = None
    while True:
        if state == "Decode":
            cand = []
            for s, seq, score in beams:
                t_f = now_ms()
                logits = model.next_logits_with_prefix(seq, oip, deterministic=det)
                stats["forward"] += now_ms() - t_f
                t_m = now_ms()
                mG = grammar_mask(product, s, tok.vocab_size, neg_inf=True)
                mT = type_mask(types, s, tok.vocab_size, neg_inf=True)
                stats["mask"] += now_ms() - t_m
                probs = softmax((logits + mG + mT))
                if legal_count(mG, mT) == 0:
                    state = "Shadow"; shadow = shadow_backoff(model, tok, product, [(s,seq,score)])
                    break
                for tkn in topk_indices(probs, k=min(64, legal_count(mG, mT)), deterministic=det):
                    s2 = product.transition(s, tkn)
                    if s2: cand.append((s2, seq+[tkn], score + float(torch.log(probs[tkn]))))
            if state != "Shadow":
                beams = prune_deterministic(cand, beam, tie="shortlex")
        elif state == "Shadow":
            t_b = now_ms()
            resync = next(shadow, [])
            stats["backoff"] += now_ms() - t_b
            if resync:
                # Boundary atomicity checks enforced by guarded byte spans
                beams = prune_deterministic(resync, beam, tie="shortlex"); state = "Decode"
            else:
                state = "Repair"
        if done(beams) or (now_ms()-t0) > sla_ms:
            break
    out = select_terminal(beams, tie="shortlex")
    if not out:
        return Abstain({"code":"no-valid-token/timeout", "stats":stats})
    text = tok.decode(out[1])
    if not validators_accept(text):
        rem = max(0, sla_ms - (now_ms()-t0))
        t_r = now_ms()
        text2, cert = anytime_repair(text, product, types, budget_ms=rem)
        stats["repair"] += now_ms() - t_r
        if text2 and validators_accept(text2): 
            return {"text": text2, "cert": cert, "stats": stats}
        return Abstain({"code":"validator/timeout", "stats":stats})
    return {"text": text, "stats": stats}
# === End block 4 ===