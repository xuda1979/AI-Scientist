def demo(json_schema_path, model_name):
    O = load_ontology("clinical_ontology.json")
    tau = load_tokenizer(model_name)
    A, Gamma = build_schema_automaton(json_schema_path)
    B = build_subword_automaton(Gamma, tau)
    P = build_product_automaton(A, B, tau.get_vocab(), tau.merges_dag())
    model = load_model_with_lora(model_name)
    prefix = init_oip(O, Gamma, model.hidden_size)
    log = Trace()
    for ex in examples():
        y = oip_cad_decode(model, prefix, tau, P, type_system(json_schema_path),
                           ex.prompt, beam=4, sla_ms=50)
        log.record(ex.id, y, validators=y is not Abstain)
    print_metrics(log)