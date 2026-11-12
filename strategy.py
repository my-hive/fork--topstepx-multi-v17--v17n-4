"""v17n-4 experiment 7: ZT flight-to-quality signal confirms ZN continuation."""
import numpy as np

FAMILY = "treasury-curve-flight-to-quality-confirmation"
PARAMS = {"minimum_zt_move_pts": 0.015625, "zn_stop_pts": 0.25}
T0, T1, EXIT = 30600, 32340, 39600  # 08:30, completed 08:59, 11:00 CT


def signals(panel):
    result = {s: np.zeros(len(df), dtype=np.int64) for s, df in panel.items()}
    if not {"ZT", "ZN"} <= set(panel):
        return result
    zt, zn = panel["ZT"], panel["ZN"]
    zt_move = {}
    for label, idx in zt.groupby("session", sort=False).indices.items():
        sec = zt.ct_sec.to_numpy()[idx]
        a, b = np.flatnonzero(sec == T0), np.flatnonzero(sec == T1)
        if len(a) and len(b):
            zt_move[str(label)] = zt.close.to_numpy()[idx[b[0]]] - zt.open.to_numpy()[idx[a[0]]]
    sec, close = zn.ct_sec.to_numpy(), zn.close.to_numpy()
    for label, idx in zn.groupby("session", sort=False).indices.items():
        move = zt_move.get(str(label), 0.0)
        if abs(move) < PARAMS["minimum_zt_move_pts"]:
            continue
        s = sec[idx]
        b = np.flatnonzero(s == T1)
        if not len(b):
            continue
        side, ref = int(np.sign(move)), close[idx[b[0]]]
        for j in range(b[0], len(idx)):
            if s[j] >= EXIT or side * (close[idx[j]] - ref) <= -PARAMS["zn_stop_pts"]:
                break
            result["ZN"][idx[j]] = side
    return result


def entry_risk(panel):
    return {s: np.full(len(df), 290.0) for s, df in panel.items()}
