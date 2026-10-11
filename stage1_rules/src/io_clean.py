"""Chistyy vvod (plumbing) dlya OBEIH etapov: chtenie CSV, chistka deneg/dat,
izvlechenie nomerov biletov. Bez resheniy: ni norm imen, ni aliasov, ni merzhey,
ni pravil match inga/atributsii, ni kursov — tolko mekhanika.

Granitsa otsenki: etot modul razreshyon na inference etapa 2 (vmeste s syrymi
CSV i frozen train/). Vsyo ostalnoe iz parse.py (imena, valyuty, load_data,
metki, komponenty) — rules, tolko etap 1 + offline-razmetka etapa 2.
Kod perenesen iz parse.py 1-v-1, bez izmeneniy logiki.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

TICKETS_DEFAULT = {"min_digits": 10, "tail": 10}


class QualityLog:
    """Sbornik nablyudeniy 'vsevozmozhnykh situatsiy' dlya quality_report.csv."""

    def __init__(self) -> None:
        self._rows: list[dict] = []
        self._seen_keys: set[tuple] = set()

    def add(self, check: str, key: str, count: int = 1,
            sample: str = "", dedup: bool = False) -> None:
        if dedup:
            marker = (check, key)
            if marker in self._seen_keys:
                return
            self._seen_keys.add(marker)
        self._rows.append({"check": check, "key": key,
                            "count": int(count), "sample": str(sample)[:200]})

    def summary(self) -> dict[str, int]:
        total: dict[str, int] = defaultdict(int)
        for row in self._rows:
            total[row["check"]] += row["count"]
        return dict(total)

    def to_frame(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(columns=["check", "key", "count", "sample"])
        frame = pd.DataFrame(self._rows)
        return frame.sort_values(["check", "key"]).reset_index(drop=True)


def read_csv_smart(path: Path, cfg: dict) -> tuple[pd.DataFrame, str]:
    """Prochitat CSV s pereborom kodirovok. Vozvraschaet (frame, encoding)."""
    last_error: Exception | None = None
    for encoding in cfg["input"]["encodings"]:
        try:
            return pd.read_csv(path, encoding=encoding), encoding
        except (UnicodeDecodeError, UnicodeError) as exc:
            last_error = exc
        except pd.errors.EmptyDataError:
            raise ValueError(f"{path}: pustoy fayl bez shapki — nechego sveryat")
    raise ValueError(f"{path}: ne udalos dekodirovat {cfg['input']['encodings']}: {last_error}")


def strip_string_cells(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip probelov; pustye stroki -> NA (platezh vs bilet chitaetsya po isna)."""
    frame = frame.copy()
    for col in frame.columns:
        if frame[col].dtype == object:
            frame[col] = frame[col].map(lambda v: v.strip() if isinstance(v, str) else v)
            frame[col] = frame[col].mask(frame[col] == "")
    return frame


def parse_money_value(value: object, seps: list[str]) -> float:
    """Razobrat odno denezhnoe znachenie: chisla prozrachno, stroki chistyatsya.
    Podderzhka: tysyachnye razdeliteli, desyatichnaya zapyataya, (skobki)."""
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    for sep in seps:
        text = text.replace(sep, "")
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    text = re.sub(r"[^0-9,.\-]", "", text)
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    try:
        number = float(text)
    except ValueError:
        return float("nan")
    return -number if negative else number


def parse_money_series(series: pd.Series, cfg: dict, qlog: QualityLog,
                       colname: str) -> pd.Series:
    # pandas 3 may infer StringDtype for CSV text columns. Treat only actual
    # numeric dtypes as already parsed; extension string dtypes still need the
    # same locale-aware parser as object columns.
    if pd.api.types.is_numeric_dtype(series.dtype):
        return series.astype(float)
    parsed = series.map(lambda v: parse_money_value(v, cfg["input"]["thousands_separators"]))
    bad = int(parsed.isna().sum() - series.isna().sum())
    if bad:
        qlog.add("unparseable_money", colname, bad,
                 str(series[parsed.isna() & series.notna()].iloc[0]), dedup=True)
    return parsed


def strip_timezone(series: pd.Series, cfg: dict, qlog: QualityLog,
                   colname: str) -> pd.Series:
    """Snyat chasovye poyasa (stennoe vremya), chtoby naive/aware ne sravnilis crash."""
    try:
        tz = series.dt.tz
    except (AttributeError, TypeError):
        return series
    if tz is not None and cfg.get("dates", {}).get("normalize_tz", True):
        if not series.notna().any():
            return series.dt.tz_localize(None)  # tolko musor -> NaT, logirovat nechego
        qlog.add("tz_normalized", colname, int(series.notna().sum()),
                 str(series[series.notna()].iloc[0]), dedup=True)
        return series.dt.tz_localize(None)
    return series


def parse_date_series(series: pd.Series, cfg: dict, qlog: QualityLog,
                      colname: str) -> pd.Series:
    """Parsit daty po spisku formatov, ostatok — svobodnym parserom.
    Vozvraschaet datetime64; neraspoznannoe -> NaT + zapis v quality."""
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    remaining = series.notna()
    for fmt in cfg["input"]["date_formats"]:
        if not remaining.any():
            break
        try:
            attempt = pd.to_datetime(series[remaining], format=fmt, errors="coerce")
        except (ValueError, TypeError):
            continue
        attempt = strip_timezone(attempt, cfg, qlog, colname)
        hit = attempt.notna()
        parsed.loc[attempt[hit].index] = attempt[hit]
        remaining = remaining & parsed.isna()
    if remaining.any():
        # utc=True: smeshannye naive/aware privedutsya k edinyy shkale, zatem snimem zonu.
        attempt = pd.to_datetime(series[remaining], errors="coerce", utc=True)
        attempt = strip_timezone(attempt, cfg, qlog, colname)
        parsed.loc[attempt[attempt.notna()].index] = attempt[attempt.notna()]
        remaining = remaining & parsed.isna()
    if remaining.any():
        qlog.add("unparseable_date", colname, int(remaining.sum()),
                 str(series[remaining].iloc[0]), dedup=True)
    return parsed


def ticket_numbers(value: object, cfg: dict | None = None) -> set[str]:
    """Vernut khvosty nomerov biletov (format 1C): tail tsifr ot dlinnykh kuskov."""
    conf = (cfg or {"tickets": TICKETS_DEFAULT})["tickets"]
    if pd.isna(value):
        return set()
    return {d[-conf["tail"]:] for d in re.findall(r"\d+", str(value))
            if len(d) >= conf["min_digits"]}
