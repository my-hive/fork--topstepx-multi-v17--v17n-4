"""v17n-4 experiment 6: fade morning Treasury move in midday dealer-liquidity window."""
import numpy as np

FAMILY = "treasury-midday-dealer-liquidity-reversal"
PARAMS = {"minimum_move_pts": 0.0625, "stop_pts": 0.25}
T0, T1, EXIT = 34200, 39540, 46800  # 09:30, completed 10:59, 13:00 CT


def signals(panel):
    result = {}
    for symbol, df in panel.items():
        out = np.zeros(len(df), dtype=np.int64)
        if symbol in ("ZN", "ZF"):
            o, c, sec = df.open.to_numpy(), df.close.to_numpy(), df.ct_sec.to_numpy()
            for idx in df.groupby("session", sort=False).indices.values():
                s = sec[idx]
                a, b = np.flatnonzero(s == T0), np.flatnonzero(s == T1)
                if not len(a) or not len(b):
                    continue
                move = c[idx[b[0]]] - o[idx[a[0]]]
                if abs(move) < PARAMS["minimum_move_pts"]:
                    continue
                side, ref = -int(np.sign(move)), c[idx[b[0]]]
                for j in range(b[0], len(idx)):
                    if s[j] >= EXIT or side * (c[idx[j]] - ref) <= -PARAMS["stop_pts"]:
                        break
                    out[idx[j]] = side
        result[symbol] = out
    return result


def entry_risk(panel):
    return {s: np.full(len(df), 1000 * PARAMS["stop_pts"] + 40.0) for s, df in panel.items()}
