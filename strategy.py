"""v17n-4 experiment 3: carry overnight Treasury direction through US session."""
import numpy as np

FAMILY = "rates-us-overnight-carry"
PARAMS = {"min_move_pts": 0.046875, "risk_dollars": 300}
START, FLAT = 30600, 54540  # 08:30 CT cash open, evaluator forced flat 15:10


def signals(panel):
    result = {}
    for symbol, df in panel.items():
        out = np.zeros(len(df), dtype=np.int64)
        if symbol in ("ZN", "ZF"):
            sec, close = df.ct_sec.to_numpy(), df.close.to_numpy()
            previous_close = None
            for idx in df.groupby("session", sort=False).indices.values():
                s = sec[idx]
                pre = np.flatnonzero(s == START - 60)
                if previous_close is not None and len(pre):
                    move = close[idx[pre[0]]] - previous_close
                    if abs(move) >= PARAMS["min_move_pts"]:
                        out[idx[(s >= START) & (s < FLAT)]] = int(np.sign(move))
                previous_close = close[idx[-1]]
        result[symbol] = out
    return result


def entry_risk(panel):
    return {s: np.full(len(df), PARAMS["risk_dollars"], dtype=float) for s, df in panel.items()}
