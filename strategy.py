"""v17n-4 experiment 2: fade first-20m cash-open Treasury move, hold to 10:00 CT."""
import numpy as np

FAMILY = "treasury-cash-open-liquidity-reversal"
PARAMS = {"observe_min": 20, "min_move_pts": 0.0625, "stop_pts": 0.2}
OPEN, EXIT = 30600, 36000  # 08:30 / 10:00 CT (fixed by market structure)


def signals(panel):
    result = {}
    last = OPEN + (PARAMS["observe_min"] - 1) * 60  # last observed bar start
    for symbol, df in panel.items():
        out = np.zeros(len(df), dtype=np.int64)
        if symbol in ("ZN", "ZF"):
            o, c, sec = df.open.to_numpy(), df.close.to_numpy(), df.ct_sec.to_numpy()
            for idx in df.groupby("session", sort=False).indices.values():
                s = sec[idx]
                a, b = np.flatnonzero(s == OPEN), np.flatnonzero(s == last)
                if not len(a) or not len(b) or b[0] - a[0] != PARAMS["observe_min"] - 1:
                    continue
                move = c[idx[b[0]]] - o[idx[a[0]]]
                if abs(move) < PARAMS["min_move_pts"]:
                    continue
                side, ref = -int(np.sign(move)), c[idx[b[0]]]
                for j in range(b[0], len(idx)):
                    # stop on completed-bar close; exit fills next open
                    if s[j] >= EXIT or side * (c[idx[j]] - ref) <= -PARAMS["stop_pts"]:
                        break
                    out[idx[j]] = side
        result[symbol] = out
    return result


def entry_risk(panel):
    # stop distance in dollars (1000/pt for ZN, ZF) plus bar-gap allowance
    return {s: np.full(len(df), 1000 * PARAMS["stop_pts"] + 50.0) for s, df in panel.items()}
