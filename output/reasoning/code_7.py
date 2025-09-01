def anytime_repair(x, fsa, types, t_budget_ms):
    y_best, cost_best = None, float('inf')
    start = now_ms()
    astar = init_astar(fsa, types, x)
    while now_ms() - start < t_budget_ms and not astar.empty():
        cand, g, h = astar.pop()  # best-first on g+h
        if g+h >= cost_best: continue
        if is_feasible(cand, fsa, types) and semantic_validator(cand):
            y_best, cost_best = cand, g
            continue
        for y_next, step_cost in expand(cand):
            astar.push(y_next, g+step_cost, heuristic(y_next))
    if y_best is not None:
        return y_best
    return ilp_with_timeout(x, fsa, types, t_budget_ms - (now_ms()-start))