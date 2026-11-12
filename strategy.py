"""v17n-4 experiment 1: rates month-end duration-extension demand."""
import numpy as np
import pandas as pd

FAMILY = "rates-month-end-duration-extension"
PARAMS = {"month_end_sessions": 3, "entry_ct_sec": 43200}


def signals(panel):
    """Long intermediate Treasury futures late in final 3 observed sessions/month."""
    result = {}
    for symbol, df in panel.items():
        out = np.zeros(len(df), dtype=np.int64)
        if symbol not in ("ZN", "ZF"):
            result[symbol] = out
            continue
        sessions = pd.to_datetime(df["session"].astype(str).str[:10])
        session_days = pd.Index(sessions.unique()).sort_values()
        is_final = np.zeros(len(df), dtype=bool)
        for _, days in pd.Series(session_days, index=session_days).groupby(session_days.to_period("M")):
            last = set(days.iloc[-PARAMS["month_end_sessions"]:])
            is_final |= sessions.isin(last).to_numpy()
        out[is_final & (df.ct_sec.to_numpy() >= PARAMS["entry_ct_sec"])] = 1
        result[symbol] = out
    return result


def entry_risk(panel):
    return {symbol: np.full(len(df), 300.0) for symbol, df in panel.items()}
