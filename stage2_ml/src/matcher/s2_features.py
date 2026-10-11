"""Pure Stage 2 pair features and mechanical component helpers.

Inference-safe: feature generation depends only on raw pair columns plus the
shared mechanical ticket parser. No Stage 1 reconciliation/identity rules.
"""
from __future__ import annotations

import re

import pandas as pd

import io_clean as C


def toks(text: object) -> set[str]:
    return set(re.findall(r"[A-ZА-Я0-9]{2,}", str(text).upper()))


def doc_toks(text: object) -> set[str]:
    return set(re.findall(r"[A-ZА-Я]{2,}-[A-ZА-Я0-9/-]+", str(text).upper()))


def ticket_features(df: pd.DataFrame) -> pd.DataFrame:
    ta = df["a_tickets"].map(lambda v: C.ticket_numbers(v, None))
    te = df["e_tickets"].map(lambda v: C.ticket_numbers(v, None))
    out = pd.DataFrame(index=df.index)
    out["tail_any"] = [(1 if a & e else 0) for a, e in zip(ta, te)]
    out["tail_frac"] = [len(a & e) / max(1, len(a | e))
                        for a, e in zip(ta, te)]
    out["amt_diff"] = (df["a_amount"] - df["e_amount"]).abs()
    out["amt_eq"] = ((df["a_amount"] - df["e_amount"]).abs() < 0.01).astype(int)
    da = pd.to_datetime(df["a_date"], errors="coerce")
    de = pd.to_datetime(df["e_date"], errors="coerce")
    out["date_gap"] = (da - de).abs().dt.total_seconds().fillna(-1) / 86400.0
    out["same_agent_raw"] = (df["a_agent"].astype(str)
                             == df["e_agent"].astype(str)).astype(int)
    for kind in ("выкуп", "возврат", "войд", "оплата"):
        out[f"ek_{kind}"] = (df["e_kind"].astype(str) == kind).astype(int)
    xa = df["a_text"].map(toks)
    xe = df["e_text"].map(toks)
    out["text_jac"] = [len(a & e) / max(1, len(a | e))
                       for a, e in zip(xa, xe)]
    out["n_a_tickets"] = df["a_tickets"].astype(str).str.count(r"\|") + 1
    return out.fillna(-1)


def pay_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["amt_diff"] = (df["a_amount"] - df["e_amount"]).abs()
    out["amt_eq"] = ((df["a_amount"] - df["e_amount"]).abs() < 0.01).astype(int)
    da = pd.to_datetime(df["a_date"], errors="coerce")
    de = pd.to_datetime(df["e_date"], errors="coerce")
    out["date_gap"] = (da - de).abs().dt.total_seconds().fillna(-1) / 86400.0
    out["same_agent_raw"] = (df["a_agent"].astype(str)
                             == df["e_agent"].astype(str)).astype(int)
    xa = df["a_text"].map(toks)
    xe = df["e_text"].map(toks)
    out["text_jac"] = [len(a & e) / max(1, len(a | e))
                       for a, e in zip(xa, xe)]
    da_ = df["a_text"].map(doc_toks)
    de_ = df["e_text"].map(doc_toks)
    out["doc_any"] = [(1 if a & e else 0) for a, e in zip(da_, de_)]
    return out.fillna(-1)


def union_find(n: int, edges: list[tuple[int, int]]) -> list[int]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for left, right in edges:
        root_l, root_r = find(left), find(right)
        if root_l != root_r:
            parent[root_r] = root_l
    return [find(i) for i in range(n)]
