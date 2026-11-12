"""v17n-4 experiment 5: Treasury continuation after scheduled inflation-release window."""
import numpy as np
import pandas as pd

FAMILY = "us-cpi-window-volatility-breakout"
PARAMS = {"minimum_move_pts": 0.046875, "stop_pts": 0.25}
T0, T1, EXIT = 30600, 32340, 37800  # 08:30, completed 08:59, 10:30 CT


def signals(panel):
    result = {}
    for symbol, df in panel.items():
        out = np.zeros(len(df), dtype=np.int64)
        if symbol in ("ZN", "ZF"):
            o, c, sec = df.open.to_numpy(), df.close.to_numpy(), df.ct_sec.to_numpy()
            for label, idx in df.groupby("session", sort=False).indices.items():
                if not 10 <= pd.Timestamp(label).day <= 14:
                    continue
                s = sec[idx]
                a, b = np.flatnonzero(s == T0), np.flatnonzero(s == T1)
                if not len(a) or not len(b):
                    continue
                move = c[idx[b[0]]] - o[idx[a[0]]]
                if abs(move) < PARAMS["minimum_move_pts"]:
                    continue
                side, ref = int(np.sign(move)), c[idx[b[0]]]
                for j in range(b[0], len(idx)):
                    if s[j] >= EXIT or side * (c[idx[j]] - ref) <= -PARAMS["stop_pts"]:
                        break
                    out[idx[j]] = side
        result[symbol] = out
    return result


def entry_risk(panel):
    return {s: np.full(len(df), 1000 * PARAMS["stop_pts"] + 40.0) for s, df in panel.items()}
