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