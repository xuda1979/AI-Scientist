import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from poker_ai.rules.side_pots import distribute_showdown


def test_odd_chip_moves_left_of_button():
    """The odd chip should go to the first player left of the button."""
    pots = [(5, [0, 1])]
    ranks = {0: 1, 1: 1}
    assert distribute_showdown(pots, ranks, button=0) == {0: 2, 1: 3}


def test_default_order_without_button():
    """Fallback ordering is ascending seat number when no button is given."""
    pots = [(5, [0, 1])]
    ranks = {0: 1, 1: 1}
    assert distribute_showdown(pots, ranks) == {0: 3, 1: 2}

