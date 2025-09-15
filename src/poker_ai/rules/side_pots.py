from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class Contribution:
    seat: int
    amount: int
    active: bool  # False if player has folded (not eligible for further pots)

def build_side_pots(contribs: List[Contribution]) -> List[Tuple[int, List[int]]]:
    """
    Compute main + side pots for multiway all-ins with folds.
    Returns a list of (pot_amount, eligible_seats).
    """
    # Filter zero contributions
    contribs = [c for c in contribs if c.amount > 0]
    if not contribs:
        return []
    # Sort unique contribution "levels"
    levels = sorted({c.amount for c in contribs})
    pots: List[Tuple[int, List[int]]] = []
    prev = 0
    for lvl in levels:
        delta = lvl - prev
        if delta <= 0:
            prev = lvl
            continue
        # Seats that have at least this much in
        eligible = [c.seat for c in contribs if c.amount >= lvl and c.active]
        # Everyone (active or not) pays delta if they put >= lvl
        payers = [c.seat for c in contribs if c.amount >= lvl]
        pot_amt = delta * len(payers)
        if pot_amt > 0 and eligible:
            pots.append((pot_amt, eligible))
        prev = lvl
    return pots

def distribute_showdown(pots: List[Tuple[int, List[int]]],
                        ranks: Dict[int, int]) -> Dict[int, int]:
    """
    Split pots according to hand ranks (lower rank value = stronger hand).
    Only seats listed in each pot's eligibility can win that pot.
    Remainders from integer division are distributed by seat order (stable).
    """
    payouts = {seat: 0 for seat in ranks}
    for pot_amt, eligible in pots:
        # Find best rank among eligible seats
        best = min((ranks[s], s) for s in eligible if s in ranks)
        best_rank = best[0]
        winners = [s for s in eligible if ranks.get(s, 10**9) == best_rank]
        share, rem = divmod(pot_amt, len(winners))
        for s in winners:
            payouts[s] += share
        # Distribute remainders deterministically by seat order
        for s in sorted(winners)[:rem]:
            payouts[s] += 1
    return payouts
