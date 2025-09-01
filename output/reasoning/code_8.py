def contrastive_loss(P, pos, neg, ontology, temperature=0.07):
    z = normalize(P.embeddings())  # prefix concept anchors
    pos_z = normalize(ontology.embed(pos))
    neg_z = normalize(ontology.embed(neg))
    logits_pos = (z @ pos_z.T) / temperature
    logits_neg = (z @ neg_z.T) / temperature
    labels = torch.cat([torch.ones_like(logits_pos), torch.zeros_like(logits_neg)], dim=1)
    logits = torch.cat([logits_pos, logits_neg], dim=1)
    return bce_with_logits(logits, labels)