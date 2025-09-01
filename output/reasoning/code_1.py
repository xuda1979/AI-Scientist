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