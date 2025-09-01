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