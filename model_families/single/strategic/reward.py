"""Only the authoritative realized ledger defines reward."""
import math
from ..accounting import realized_ledger


class LedgerReward:
    def __init__(self):
        self.previous = 0.
        self.total = 0.

    def advance(self, sim, orders, rows, terminal=False):
        ledger = realized_ledger(sim, orders, rows)
        value = ledger["realized_profit"]
        reward = value-self.previous
        if not math.isfinite(reward):
            raise RuntimeError("nonfinite realized reward")
        self.previous = value
        self.total += reward
        if terminal and (ledger["unfinished_committed"] or
                         not math.isclose(self.total, value, rel_tol=1e-10, abs_tol=1e-7)):
            raise RuntimeError("full cooldown/reward reconciliation failed")
        return reward, ledger
