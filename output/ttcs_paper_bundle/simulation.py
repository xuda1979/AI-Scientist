# Auto-generated from LaTeX code blocks; consolidate all simulation here.
# === Begin extracted block 1 ===
# Uniform Tick API and a simple budgeted scheduler.

def marginal_gain_from_uncertainty(u):
    """Default mapping from uncertainty to marginal gain.
    Falls back to 0.0 if u is not numeric.
    """
    try:
        return float(u)
    except Exception:
        return 0.0


def fuse_outputs(outputs):
    """Default fusion of agent outputs.
    Returns the first non-None output; None if all are None.
    Replace with domain-specific fusion as needed.
    """
    for o in outputs:
        if o is not None:
            return o
    return None


class Agent:
    def __init__(self, module, cost_per_tick=1.0):
        self.m = module
        self.cost = cost_per_tick

    def bid(self, shared_state):
        # uncertainty -> marginal gain (can be learned or calibrated map)
        u = self.m.uncertainty(shared_state)
        gain = marginal_gain_from_uncertainty(u)
        def step():
            self.m.one_tick(shared_state)
        return gain, step


def budgeted_inference(shared_state, agents, B, lam=0.0):
    spent = 0.0
    while spent < B:
        bids = []
        for a in agents:
            g, step = a.bid(shared_state)
            # Net surplus: gain minus "price" of compute
            surplus = g - lam * a.cost
            if surplus > 0:
                bids.append((surplus, g, a.cost, step))
        if not bids:
            break
        bids.sort(reverse=True, key=lambda t: t[0])
        _, g, c, step = bids[0]
        step()                 # consume one tick at best bidder
        spent += c
    return fuse_outputs([a.m.output(shared_state) for a in agents])
# === End block 1 ===
