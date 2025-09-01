def anytime_repair(x, P, T, budget_ms):
    start, best_y, best_cost = now_ms(), None, float('inf')
    open_set = init_astar(x, P, T)
    while now_ms() - start < budget_ms and not open_set.empty():
        y, g, h = open_set.pop_best()  # minimize g+h; tie-break shortlex
        if g+h >= best_cost: continue
        if feasible(y, P, T) and semantic_validators(y):
            best_y, best_cost = y, g; continue
        for y2, step_cost in expand_with_types(y, P, T):
            open_set.push(y2, g+step_cost, heuristic(y2, P, T))
    if best_y: return best_y
    remaining = max(0, budget_ms - (now_ms()-start))
    return ilp_with_timeout(x, P, T, timeout_ms=remaining)