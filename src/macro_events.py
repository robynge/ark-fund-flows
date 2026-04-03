"""Macro event registry and data filtering.

Each event is manually defined with an ID, date range, and description.
Treatment methods (exclude dates or add dummy variables) are applied as
composable DataFrame transforms.
"""
import pandas as pd
import numpy as np

EVENTS = [
    {
        "id": "covid",
        "name": "COVID-19 Crash & Recovery",
        "start": "2020-02-01",
        "end": "2020-06-30",
        "description": "Global pandemic crash and initial recovery rally",
    },
    {
        "id": "ukraine",
        "name": "Russia-Ukraine War",
        "start": "2022-02-24",
        "end": "2022-04-30",
        "description": "Russian invasion of Ukraine; energy/commodity shock",
    },
    {
        "id": "fed_hikes",
        "name": "Fed Rate Hike Cycle",
        "start": "2022-03-17",
        "end": "2023-07-26",
        "description": "Fed funds rate from 0.25% to 5.50%; growth-to-value rotation",
    },
    {
        "id": "banking_crisis",
        "name": "Regional Banking Crisis",
        "start": "2023-03-08",
        "end": "2023-05-01",
        "description": "SVB, Signature, First Republic failures; contagion fears",
    },
    {
        "id": "arkb_approval",
        "name": "Spot Bitcoin ETF Approval",
        "start": "2024-01-10",
        "end": "2024-02-28",
        "description": "SEC approves spot Bitcoin ETFs; massive inflows to Bitcoin ETFs",
    },
]


def get_event(event_id: str) -> dict:
    """Look up a single event by ID."""
    for e in EVENTS:
        if e["id"] == event_id:
            return e
    raise ValueError(f"Unknown event ID: {event_id}. "
                     f"Available: {[e['id'] for e in EVENTS]}")


def get_event_ids() -> list[str]:
    """Return all registered event IDs."""
    return [e["id"] for e in EVENTS]


def event_mask(df: pd.DataFrame, event_id: str,
               date_col: str = "Date") -> pd.Series:
    """Return a boolean mask: True for rows inside the event window."""
    event = get_event(event_id)
    start = pd.Timestamp(event["start"])
    end = pd.Timestamp(event["end"])
    return (df[date_col] >= start) & (df[date_col] <= end)


def exclude_events(df: pd.DataFrame, event_ids: list[str],
                   date_col: str = "Date") -> pd.DataFrame:
    """Remove rows falling within any of the specified event windows."""
    mask = pd.Series(False, index=df.index)
    for eid in event_ids:
        mask = mask | event_mask(df, eid, date_col)
    return df[~mask].copy()


def add_event_dummies(df: pd.DataFrame, event_ids: list[str],
                      date_col: str = "Date") -> pd.DataFrame:
    """Add binary dummy columns for each specified event (1 = inside window)."""
    df = df.copy()
    for eid in event_ids:
        col_name = f"event_{eid}"
        df[col_name] = event_mask(df, eid, date_col).astype(int)
    return df


def apply_event_treatment(df: pd.DataFrame, event_ids: list[str],
                          method: str = "exclude",
                          date_col: str = "Date") -> pd.DataFrame:
    """Apply event treatment to DataFrame.

    Parameters:
        event_ids: list of event IDs to treat
        method: "exclude" (drop rows), "dummy" (add columns), or "both"
        date_col: name of the date column
    """
    if method == "exclude":
        return exclude_events(df, event_ids, date_col)
    elif method == "dummy":
        return add_event_dummies(df, event_ids, date_col)
    elif method == "both":
        df = add_event_dummies(df, event_ids, date_col)
        return exclude_events(df, event_ids, date_col)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'exclude', 'dummy', or 'both'.")
