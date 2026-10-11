"""Obschiy modul pravil etapa 1 (poverh io_clean).

Vladeltsy: dvizhok reconcile2.py (etap 1) i offline-razmetka etapa 2
(matcher train-fayly). Na inference etapa 2 — ZAPRESCHYON (sm. EVAL_PROTOCOL.md):
tam tolko io_clean + syrye CSV + frozen train/.
Pravilo: lyubaya pravka zdes affektit OBA potrebitelya — posle pravki progon:
  1) python3 reconcile2.py + sverka golden-md5 + pytest tests/ -q (cwd stage1_rules/src)
  2) matcher reproduce.py + diff_discrepancies.py + sverka tsifr s EVAL.md
Perenos 1-v-1, bez izmeneniy logiki: parsyatsya te zhe kolonki, te zhe defolty.

Soderzhit: config (load_config/validate), imena (exact+alias, bez fuzzy-sliyaniya —
ono v dvizhke), valyuty, load_data (freymy + agent_norm/ticket_set/sum_1c/sum_etm),
podpisi agentov (canonical_labels), svyaznye komponenty biletov (ticket_components).
Chistyy vvod (QualityLog, CSV, dengi/daty, khvosty biletov) — sm. io_clean.
"""
from __future__ import annotations

import difflib
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import pandas as pd

from io_clean import (  # chistyy vvod (plumbing, obschiy dlya obeih etapov)
    QualityLog,
    TICKETS_DEFAULT,
    parse_date_series,
    parse_money_series,
    parse_money_value,
    read_csv_smart,
    strip_string_cells,
    strip_timezone,
    ticket_numbers,
)

try:
    import yaml
except ImportError:  # PyYAML ne obyazatelen: bez nego — vstroennye defolty
    yaml = None

HERE = Path(__file__).resolve().parent
CONFIG_NAME = "reconcile.yaml"

DEFAULT_CONFIG: dict = {
    "tolerance": 0.01,
    "input": {
        "encodings": ["utf-8-sig", "utf-8", "cp1251"],
        "strip_strings": True,
        "date_formats": [
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%d.%m.%Y",
            "%d.%m.%Y %H:%M:%S",
            "%d/%m/%Y",
            "%Y/%m/%d",
        ],
        "thousands_separators": [" ", " ", "'", "’"],
    },
    "names": {
        "stopwords": [
            "осоо", "ooo", "ocoo", "ип",
            "зао", "oao", "оао", "llc", "ltd", "гп", "чп", "чуп",
        ],
        # Latin -> Cyrillic homoglyph fold (primenyaetsya k nospace-klyuchu).
        # Samaya chastaya 'opechatka v 1 simvol': latinskaya bukva vmesto kirillicheskoy.
        "homoglyphs": {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с",
                       "x": "х", "m": "м", "k": "к", "t": "т", "b": "в", "h": "н"},
        # Fuzzy-sloy (difflib, tolko dlya podskazok i strogogo auto-merge reestra).
        "fuzzy_suggest_threshold": 0.90,
        "fuzzy_auto_threshold": 0.94,
        # auto_merge=True: reestr sam skлеиваetsya po nospace/homoglyph/strogomu fuzzy.
        # Dengi (akty<->ETM) vsegda strogie: exact+alias, mёrdzh imen na nikh ne vliyaet.
        "auto_merge": True,
        # subset_auto_merge=True (v2): korotkoe imya partii — token-podmnozhestvo
        # rovno odnogo agenta + obschie bilety -> avtomerzh (sm. resolve_registry_party).
        "subset_auto_merge": True,
        # enabled=False: tolko exact+alias (rezhim pervoy versii; dlya A/B-proverok).
        "enabled": True,
    },
    "status_policy": {
        "reissued_markers": ["переиздан"],
        "drop_if_reissued": ["черновик"],
        "always_drop": [],
    },
    "tickets": TICKETS_DEFAULT,  # edinstvennyy istochnik (sm. io_clean)
    "payments": {
        "date_window_days": 120,
        "review_gap_days": 45,
        "prefer_doc_match": True,
        "doc_pattern": r"ЦБ-С\d+",
    },
    "void_kinds": ["войд", "void"],
    "sale_kinds": ["выкуп"],
    "dates": {
        # Smeshannye chasovye poyasa (naive + +06:00) padali s TypeError:
        # privodim vse k naive (stennoe vremya), sluchai — v quality.
        "normalize_tz": True,
        # Data vne [min_period - margin, max_period_end + margin] — opechatka.
        "range_margin_days": 15,
        # Voyd do prodazhi: gap <= edge schitaem kraem okna (dekabr), bolshe — alarm.
        # Flaguyutsya VSE sluchai s velichinoy gapa (reshenie: sortirovka otdelit).
        "edge_days": 45,
        "check_balance_chain": True,
        "balance_tolerance": 0.01,
    },
    "currencies": {
        "base": "KGS",
        "resolved_column": "amount_kgs",
        "amount_column": "amount",
        "currency_column": "currency",
        "rate_columns": {
            "USD": "rate_usd",
            "EUR": "rate_eur",
            "RUB": "rate_rub",
            "KZT": "rate_kzt",
        },
        "fallback_rates": {},
    },
    "agents": {"aliases": {}},
    "labels": {
        # Kanonicheskaya podpis agenta v otchetakh: acts_first (povedenie v1),
        # alternativa most_frequent (chashche vsego vstrechayuschayasya syraya podpis).
        "source": "acts_first",
    },
    "comments": {"void_remainder_note": False},
    "v3": {
        # Mikroruly atributsii (adjudication 150): R1/R3a/R3b.
        # enabled=False: rezhim v2 (tolko prefix-mepping source; dlya A/B-proverok).
        "enabled": True,
        # CONV_TOL dlya reg_eq_conv: kursy v stroke okrugleny (S110 +0.11, S119 +0.19).
        "conv_tol": 2.0,
        # R1: srabatyvaet tolko na znachimyh summah (genuine-meloch S101-S105 < 200).
        "r1_min_diff": 500.0,
        # RA (agent-klass): porog znachimosti (meloch ne atributiruem).
        "ra_min_diff": 500.0,
        # R3: bliznets = ta zhe summa+vid, <= twin_window_days (nablyudennyy max ~4d).
        "twin_window_days": 7,
        # Data vygruzki: "auto" = max data acts+etm (rekomenduetsya: nichego ne
        # nuzhno menyat kazhdyy mesyats); mozhno yavnuyu datu "2026-07-31".
        # Molozhe data_end - fresh_days schitaetsya lagom, ne v3-sluchaem.
        "data_end": "auto",
        "fresh_days": 31,
    },
    "output": {"quality_report": "quality_report.csv"},
}

REQUIRED_ACTS = {"folder", "period_start", "period_end", "saldo_start",
                 "saldo_end", "date", "doc", "ticket_cell", "debet", "credit"}
REQUIRED_ETM = {"agent", "txn_id", "date", "kind", "tickets",
                "amount_kgs", "balance_after"}
REQUIRED_REG = {"date", "employee", "party", "tickets"}

# Esli resolved_column net — trebuem syrye kolonki dlya konversii.
REQUIRED_ETM_NOCONV = {"agent", "txn_id", "date", "kind", "tickets",
                       "amount", "balance_after"}

# ---------------------------------------------------------------- config ---

def deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def validate_config(user: dict, defaults: dict, path: str = "") -> list[str]:
    """Sverka polzovatelskogo konfiga so skhemoy (DEFAULT_CONFIG).
    Filosofiya 'nikogda ne padaem': neizvestnyy klyuch ignoriruetsya,
    nevernyy tip zamenyaetsya defoltom — vse s WARNING, a ne isklyucheniem."""
    warnings: list[str] = []
    for key, value in (user or {}).items():
        dotted = f"{path}.{key}" if path else str(key)
        if not isinstance(defaults, dict) or key not in defaults:
            warnings.append(f"unknown config key '{dotted}' (ignored)")
            continue
        expected = defaults[key]
        if isinstance(value, dict) and isinstance(expected, dict):
            warnings.extend(validate_config(value, expected, dotted))
        elif not _type_matches(value, expected):
            warnings.append(
                f"config key '{dotted}': ozhidalsya {type(expected).__name__}, "
                f"poluchen {type(value).__name__} — ispolzuyu default")
    return warnings


def _type_matches(value: object, expected: object) -> bool:
    if isinstance(expected, bool) or isinstance(value, bool):
        return isinstance(value, bool) and isinstance(expected, bool)
    if isinstance(expected, float):
        return isinstance(value, (int, float))
    if isinstance(expected, int):
        return isinstance(value, int)
    if isinstance(expected, str):
        return isinstance(value, str)
    if isinstance(expected, list):
        return isinstance(value, list)
    return True


def validated_merge(base: dict, override: dict) -> tuple[dict, list[str]]:
    """Deep-merge tolko izvestnykh klyuchey s vernym tipom + spisok warnings."""
    warnings = validate_config(override, base)
    cleaned = _strip_invalid(override or {}, base)
    return deep_merge(base, cleaned), warnings


def _strip_invalid(user: dict, defaults: dict) -> dict:
    cleaned: dict = {}
    for key, value in (user or {}).items():
        if not isinstance(defaults, dict) or key not in defaults:
            continue
        expected = defaults[key]
        if isinstance(value, dict) and isinstance(expected, dict):
            cleaned[key] = _strip_invalid(value, expected)
        elif _type_matches(value, expected):
            cleaned[key] = deepcopy(value)
    return cleaned


def load_config(path: Path | None) -> dict:
    """Otdat dict konfiga: yavno ukazannyy fayl, libo reconcile.yaml ryadom,
    libo stage1 input/ (novaya raskladka finolimp), libo vstroennye defolty.
    Otsutstvie PyYAML — ne oshibka, a defolty."""
    candidates = ([Path(path)] if path else []) + [
        HERE / CONFIG_NAME,
        HERE.parent / "input" / CONFIG_NAME,
    ]
    cfg_path = next((c for c in candidates if Path(c).exists()), None)
    if cfg_path and Path(cfg_path).exists():
        if yaml is None:
            print(f"WARNING: {cfg_path} nayden, no PyYAML otsutstvuet — ispolzuyu defolty")
            return deepcopy(DEFAULT_CONFIG)
        with open(cfg_path, encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
        cfg, warnings = validated_merge(DEFAULT_CONFIG, user)
        for warning in warnings:
            print(f"WARNING (config): {warning}")
        source = (cfg.get("labels") or {}).get("source")
        if source not in ("acts_first", "most_frequent"):
            print(f"WARNING (config): labels.source='{source}' neizvesten — ispolzuyu acts_first")
            cfg["labels"]["source"] = "acts_first"
        return cfg
    return deepcopy(DEFAULT_CONFIG)


# -------------------------------------------------------------- quality ---
# (QualityLog — sm. io_clean: obschiy dlya obeih etapov)


# ---------------------------------------------------------------- input ---
# (read_csv_smart / strip_string_cells / parse_money_value — sm. io_clean)


PAY_RATE_COL = {"usd": "rate_usd", "eur": "rate_eur", "rub": "rate_rub",
                "kzt": "rate_kzt"}
PAY_KGS_WORDS = ("kgs", "кгс", "сом", "сомов")


def resolve_data_end(cfg: dict, acts=None, etm=None, qlog=None):
    """Data vygruzki dlya fresh-gardov: 'auto' = max date_norm acts+etm,
    yavnaya data — kak est. None + warning esli vychislit nelzya
    (toggda v3 svezhestnie gardy propuskayutsya — never-fail)."""
    try:
        v3 = (cfg or {}).get("v3", {}) or {}
        raw = str(v3.get("data_end", "auto")).strip()
    except (AttributeError, ValueError):
        raw = "auto"
    if raw.lower() != "auto":
        try:
            return pd.Timestamp(raw)
        except (ValueError, TypeError):
            if qlog is not None:
                qlog.add("v3_no_data_end", "config", 1, raw, dedup=True)
            return None
    dates = []
    for frame in (acts, etm):
        if frame is not None and "date_norm" in frame.columns:
            mx = frame["date_norm"].max()
            if pd.notna(mx):
                dates.append(mx)
    if not dates:
        if qlog is not None:
            qlog.add("v3_no_data_end", "auto", 1, "net dat v acts/etm", dedup=True)
        return None
    return pd.Timestamp(max(dates))


def parse_pay_kgs_total(pay_cell: object, rates) -> float:
    """Konvertirovat pay_cell reestra v KGS po kursam stroki; NaN esli hotya by
    odin chunk ne den'gi (ssylka na bilet bez valyuty, neizvestnaya valyuta,
    net kursa). Pravila iz adjudication S110/S112/S119: vse chunki summiruyutsya
    (vklyuchaya sf), minus = vozvrat/shtraf (otritsatel'nyj vklad)."""
    if not isinstance(pay_cell, str) or not pay_cell.strip():
        return float("nan")
    total = 0.0
    for chunk in pay_cell.split("+"):
        chunk = re.sub(r"\([^()]*[A-Za-zА-Яа-я][^()]*\)", "", chunk)
        cur = [c for c in re.findall(r"[A-Za-zА-Яа-я]+", chunk)
               if c.lower() != "sf"]
        if not cur:
            return float("nan")
        low = [c.lower() for c in cur]
        if any(c not in PAY_RATE_COL and c not in PAY_KGS_WORDS for c in low):
            return float("nan")
        v = parse_money_value(chunk, [])
        if pd.isna(v):
            return float("nan")
        foreign = [c for c in low if c in PAY_RATE_COL]
        if foreign:
            if len(set(foreign)) > 1:
                return float("nan")  # v dannyh ne vstrechaetsya (zamer: 0)
            try:
                rate = float(rates[PAY_RATE_COL[foreign[0]]])
            except (KeyError, TypeError, ValueError):
                return float("nan")
            if pd.isna(rate):
                return float("nan")
            total += v * rate
        else:
            total += v
    return round(total, 2)


# (parse_money_series / strip_timezone / parse_date_series — sm. io_clean)


# ---------------------------------------------------------------- names ---

def stopwords_pattern(stopwords: list[str]) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(re.escape(w) for w in stopwords) + r")\b")


def norm_agent(value: object, cfg: dict | None = None) -> str:
    """Privesti nazvanie kontragenta k kanonicheskomu vidu dlya match rester."""
    stopwords = (cfg or DEFAULT_CONFIG)["names"]["stopwords"]
    s = str(value).strip().lower().replace("ё", "е")
    s = re.sub(r'["«»\.]', " ", s)
    s = stopwords_pattern(stopwords).sub(" ", s)
    s = re.sub(r"[\s\-]+", " ", s).strip()
    return re.sub(r"\s+", " ", s)


def apply_aliases(normed: str, cfg: dict) -> str:
    """Pereimenovaniya agentov: norm -> kanonicheskiy norm (sredina goda)."""
    aliases = (cfg.get("agents") or {}).get("aliases") or {}
    normed_aliases = {norm_agent(k, cfg): norm_agent(v, cfg) for k, v in aliases.items()}
    seen: set[str] = set()
    while normed in normed_aliases and normed not in seen:
        seen.add(normed)
        normed = normed_aliases[normed]
    return normed


def homoglyph_fold(text: str, cfg: dict) -> str:
    """Svernut latinskie omoglify v kirillitsu (tablitsa names.homoglyphs)."""
    table = (cfg.get("names") or {}).get("homoglyphs") or {}
    if not table:
        return text
    return text.translate(str.maketrans(table))


def build_name_index(core_norms: list[str], cfg: dict) -> dict:
    """Indeks klyuchey yadra (akty+ETM): exact + nospace + homoglyph.
    Kollizii (dva agenta na odin klyuch) -> None: ne mёrdzhim, flag nem observation."""
    nospace_map: dict[str, list[str]] = defaultdict(list)
    homo_map: dict[str, list[str]] = defaultdict(list)
    for normed in core_norms:
        nospace_map[normed.replace(" ", "")].append(normed)
        homo_map[homoglyph_fold(normed.replace(" ", ""), cfg)].append(normed)
    return {
        "exact": set(core_norms),
        "nospace": {k: (v[0] if len(v) == 1 else None) for k, v in nospace_map.items()},
        "homo": {k: (v[0] if len(v) == 1 else None) for k, v in homo_map.items()},
        "nospace_collisions": {k: v for k, v in nospace_map.items() if len(v) > 1},
    }


def agent_ticket_sets(acts: pd.DataFrame, etm: pd.DataFrame) -> dict[str, set[str]]:
    """Bilety kazhdogo agenta yadra (ulika dlya strogogo pravila auto-merge)."""
    tickets: dict[str, set[str]] = defaultdict(set)
    for _, row in acts.iterrows():
        tickets[row["agent_norm"]] |= set(row["ticket_set"])
    for _, row in etm.iterrows():
        tickets[row["agent_norm"]] |= set(row["ticket_set"])
    return tickets


def name_similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, left, right).ratio()


def resolve_registry_party(
    party_norm: str,
    party_tickets: set[str],
    index: dict,
    core_list: list[str],
    tickets_of: dict[str, set[str]],
    cfg: dict,
    qlog: QualityLog,
    raw_label: str,
    n_rows: int,
) -> str:
    """Razreshit normu storony reestra v normu yadra (akty+ETM).
    Urovni: exact -> alias (uzhe primenen ranee) -> nospace -> homoglyph ->
    fuzzy-strogiy (tolko pri auto_merge) -> unresolved (svoya norma, orphan).
    Kazhdoe nesovpadayuschee reshenie pishetsya v quality (nablyudaemost)."""
    ncfg = cfg.get("names") or {}
    if party_norm in index["exact"]:
        return party_norm
    if not ncfg.get("enabled", True):
        return party_norm
    nospace_key = party_norm.replace(" ", "")
    hit = index["nospace"].get(nospace_key)
    if hit is not None:
        qlog.add("registry_nospace_hit", f"{raw_label} -> {hit}", n_rows, "", dedup=True)
        return hit
    if nospace_key in index["nospace_collisions"]:
        qlog.add("ambiguous_nospace", nospace_key, n_rows,
                 " | ".join(index["nospace_collisions"][nospace_key]), dedup=True)
        return party_norm
    homo_key = homoglyph_fold(nospace_key, cfg)
    hit = index["homo"].get(homo_key)
    if hit is not None:
        qlog.add("registry_homoglyph_hit", f"{raw_label} -> {hit}", n_rows, "", dedup=True)
        return hit
    # Subset (v2): korotkoe imya partii — strogoe token-podmnozhestvo rovno
    # odnogo agenta yadra + obschie bilety. Ulika chestnaya: tickets_of
    # postroeny tolko po aktam+ETM (bez reestra) — samoperekrytie isklyucheno
    # (sovpadenie s sobstvennoy normoy uzhe srabotalo by na exact vyshe).
    if ncfg.get("subset_auto_merge", True):
        ptoks = set(party_norm.split())
        cands = [c for c in core_list if ptoks and ptoks <= set(c.split())]
        if len(cands) == 1:
            best = cands[0]
            shared_n = len(party_tickets & tickets_of.get(best, set()))
            if shared_n:
                qlog.add("registry_subset_hit",
                         f"{raw_label} -> {best}", n_rows,
                         f"tokensubset+{shared_n} obschikh biletov", dedup=True)
                return best
            qlog.add("possible_duplicate_agents", raw_label, n_rows,
                     f"tokensubset {best}, obschikh biletov net", dedup=True)
        elif len(cands) > 1:
            qlog.add("ambiguous_subset", party_norm, n_rows,
                     " | ".join(sorted(cands)[:5]), dedup=True)
    # Fuzzy: kandidatov malo (yadro ~100), difflib bez blokirovki dostatochen.
    suggest_thr = float(ncfg.get("fuzzy_suggest_threshold", 0.90))
    auto_thr = float(ncfg.get("fuzzy_auto_threshold", 0.94))
    scored = [(c, name_similarity(party_norm, c))
              for c in core_list if abs(len(c) - len(party_norm)) <= 3]
    scored = sorted(((c, s) for c, s in scored if s >= suggest_thr),
                    key=lambda t: -t[1])[:3]
    if scored:
        best, best_score = scored[0]
        unique = len(scored) == 1 or scored[1][1] < auto_thr
        shared = bool(party_tickets & tickets_of.get(best, set()))
        strict = best_score >= auto_thr and unique and shared
        if strict and ncfg.get("auto_merge", True):
            qlog.add("registry_fuzzy_hit",
                     f"{raw_label} -> {best} ({best_score:.2f}, obschie bilety)", n_rows, "", dedup=True)
            return best
        sample = "; ".join(f"{c} ({s:.2f})" for c, s in scored)
        sample += f"; obschie bilety: {'da' if shared else 'net'}"
        qlog.add("possible_duplicate_agents", raw_label, n_rows, sample, dedup=True)
    return party_norm


# (ticket_numbers — sm. io_clean: mekhanika khvostov, bez resheniy)


def payment_identifier(document: str) -> str:
    match = re.search(r"ЦБ-С\d+", str(document))
    return match.group(0) if match else str(document).strip()


def doc_ref(text: object, cfg: dict) -> str:
    """Izvlech klyuch platezha (nomer ЦБ-С) iz proizvolnogo teksta."""
    if pd.isna(text):
        return ""
    match = re.search(cfg["payments"]["doc_pattern"], str(text))
    return match.group(0) if match else ""


# -------------------------------------------------------------- currency ---

def build_rate_table(reg: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Tablica kursov po valyutam iz kolonok reestra: currency -> (date, rate)."""
    table: dict[str, pd.DataFrame] = {}
    rate_cols = cfg["currencies"]["rate_columns"]
    for currency, col in rate_cols.items():
        if col in reg.columns and "date_norm" in reg.columns:
            frame = reg.loc[reg[col].notna(), ["date_norm", col]].copy()
            frame = frame.sort_values("date_norm").reset_index(drop=True)
            if len(frame):
                table[currency.upper()] = frame
    return table


def rate_for(table: dict[str, pd.DataFrame], fallbacks: dict,
             currency: str, when: pd.Timestamp) -> float | None:
    currency = str(currency).strip().upper()
    if currency in table:
        frame = table[currency]
        past = frame[frame["date_norm"] <= when]
        if len(past):
            return float(past.iloc[-1, 1])
        return float(frame.iloc[0, 1])
    if currency in (fallbacks or {}):
        return float(fallbacks[currency])
    return None


def resolve_sum_etm(etm: pd.DataFrame, reg: pd.DataFrame, cfg: dict,
                    qlog: QualityLog) -> pd.Series:
    """Summa ETM v znake 1C. Legasy-put: gotovaya amount_kgs bez propuskov.
    Inache: amount * kurs(valyuta, data). Neizvestnaya valyuta -> NaN + quality."""
    ccfg = cfg["currencies"]
    resolved = ccfg["resolved_column"]
    if resolved in etm.columns and etm[resolved].notna().all():
        return (-etm[resolved].astype(float)).round(2)
    if ccfg["amount_column"] not in etm.columns:
        raise ValueError(f"etm: net ni {resolved}, ni {ccfg['amount_column']} dlya perescheta valyut")
    table = build_rate_table(reg, cfg)
    amounts = parse_money_series(etm[ccfg["amount_column"]], cfg, qlog, ccfg["amount_column"])
    currencies = etm[ccfg["currency_column"]].astype(str) if ccfg["currency_column"] in etm.columns else "KGS"
    base = ccfg["base"].upper()
    out = pd.Series(float("nan"), index=etm.index)
    for idx in etm.index:
        cur = str(currencies.loc[idx]).strip().upper() if not isinstance(currencies, str) else base
        if cur in ("", "NAN", "NONE"):
            cur = base
        if cur == base:
            rate = 1.0
        else:
            when = etm.loc[idx, "date_norm"] if "date_norm" in etm.columns else pd.NaT
            rate = rate_for(table, ccfg.get("fallback_rates") or {}, cur, when)
            if rate is None:
                qlog.add("unknown_currency", cur, 1, str(etm.loc[idx, "txn_id"]), dedup=True)
                continue
        out.loc[idx] = -float(amounts.loc[idx]) * float(rate)
    return out.round(2)


# ----------------------------------------------------------------- load ---

def load_data(acts_path: Path, etm_path: Path, registry_path: Path,
              cfg: dict, qlog: QualityLog
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    acts, acts_enc = read_csv_smart(acts_path, cfg)
    etm, etm_enc = read_csv_smart(etm_path, cfg)
    reg, reg_enc = read_csv_smart(registry_path, cfg)
    if acts_enc != "utf-8-sig" and acts_enc != "utf-8":
        qlog.add("nonstandard_encoding", str(acts_path), 1, acts_enc, dedup=True)
    if etm_enc != "utf-8-sig" and etm_enc != "utf-8":
        qlog.add("nonstandard_encoding", str(etm_path), 1, etm_enc, dedup=True)

    if cfg["input"]["strip_strings"]:
        acts, etm, reg = (strip_string_cells(f) for f in (acts, etm, reg))

    conv = REQUIRED_ETM_NOCONV if "amount_kgs" not in etm.columns else REQUIRED_ETM
    for name, frame, required in (
        (str(acts_path), acts, REQUIRED_ACTS),
        (str(etm_path), etm, conv),
        (str(registry_path), reg, REQUIRED_REG),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name}: missing columns: {', '.join(sorted(missing))}")

    # --- tsentralnaya chistka deneg: syrye stroki ('-25 000,00', '(33 000)', 'н/д')
    # parsyatsya ZDES, dalshe vezde float. Inache sravneniya/float() padayut
    # na strankh do togo, kak parse_money_series uspeet ikh pochistit.
    for col in ("debet", "credit", "saldo_start", "saldo_end"):
        acts[col] = parse_money_series(acts[col], cfg, qlog, f"acts.{col}")
    for col in ("amount_kgs", "amount", "balance_after"):
        if col in etm.columns:
            etm[col] = parse_money_series(etm[col], cfg, qlog, f"etm.{col}")
    for col in set(cfg.get("currencies", {}).get("rate_columns", {}).values()):
        if col in reg.columns:
            reg[col] = parse_money_series(reg[col], cfg, qlog, f"registry.{col}")

    # --- act_status: politika iz konfiga (defolt = povedenie pervoy versii) ---
    ignored_drafts: dict[str, int] = defaultdict(int)
    spol = cfg["status_policy"]
    if "act_status" in acts.columns:
        status = acts["act_status"].fillna("").astype(str).str.strip().str.lower()
        known = set(spol["reissued_markers"]) | set(spol["drop_if_reissued"]) | set(spol["always_drop"])
        for value in sorted(set(status) - {""} - known):
            qlog.add("unknown_status", value,
                     int((status == value).sum()), "", dedup=True)
        drop_always = status.isin([s.lower() for s in spol["always_drop"]])
        if int(drop_always.sum()):
            qlog.add("dropped_status", ",".join(sorted(set(status[drop_always]))),
                     int(drop_always.sum()), "")
        markers = [s.lower() for s in spol["reissued_markers"]]
        drop_names = {s.lower() for s in spol["drop_if_reissued"]}
        has_reissued = (
            acts.assign(_st=status)
            .groupby(["folder", "period_start"])["_st"]
            .transform(lambda s: bool(s.isin(markers).any()))
        )
        drop_mask = (drop_always | (has_reissued & status.isin(drop_names))).fillna(False)
        acts["_norm_tmp"] = acts["folder"].astype(str).map(lambda v: norm_agent(v, cfg))
        for norm_key, group in acts.loc[drop_mask].groupby("_norm_tmp"):
            ignored_drafts[norm_key] = int(len(group))
        acts = acts.loc[~drop_mask].copy().reset_index(drop=True)
        acts = acts.drop(columns=["_norm_tmp"], errors="ignore")

    # --- neizvestnye kind/creator: nablyudaem, ne padaem ---
    for col in ("kind", "creator"):
        if col in etm.columns:
            known_kinds = {"выкуп", "оплата", "возврат", "войд"} if col == "kind" \
                else {"etm-bot", "субагент"}
            unknown = etm.loc[~etm[col].astype(str).str.strip().str.lower().isin(known_kinds), col]
            for value in sorted(set(unknown.astype(str))):
                qlog.add(f"unknown_etm_{col}", value, int((unknown.astype(str) == value).sum()),
                         "", dedup=True)

    acts["agent_raw"] = acts["folder"].astype(str).str.strip()
    etm["agent_raw"] = etm["agent"].astype(str).str.strip()
    acts["agent_norm"] = acts["agent_raw"].map(lambda v: apply_aliases(norm_agent(v, cfg), cfg))
    etm["agent_norm"] = etm["agent_raw"].map(lambda v: apply_aliases(norm_agent(v, cfg), cfg))
    # reg["agent_norm"] stavitsya pozzhe — sloem imen (resolve_registry_party).
    acts["_aliased_from"] = acts["agent_raw"].where(
        acts["agent_raw"].map(lambda v: norm_agent(v, cfg)) != acts["agent_norm"], "")
    etm["_aliased_from"] = etm["agent_raw"].where(
        etm["agent_raw"].map(lambda v: norm_agent(v, cfg)) != etm["agent_norm"], "")

    acts["agent_label"] = acts["folder"].astype(str).str.strip()
    etm["agent_label"] = etm["agent"].astype(str).str.strip()

    acts["date_norm"] = parse_date_series(acts["date"], cfg, qlog, "acts.date")
    etm["date_norm"] = parse_date_series(etm["date"], cfg, qlog, "etm.date")
    reg["date_norm"] = parse_date_series(reg["date"], cfg, qlog, "registry.date")
    acts["period_start_norm"] = parse_date_series(acts["period_start"], cfg, qlog, "acts.period_start")
    acts["period_end_norm"] = parse_date_series(acts["period_end"], cfg, qlog, "acts.period_end")
    acts["sum_1c"] = (parse_money_series(acts["debet"], cfg, qlog, "acts.debet").fillna(0)
                       - parse_money_series(acts["credit"], cfg, qlog, "acts.credit").fillna(0)).round(2)
    neg_money = acts[(acts["debet"].fillna(0) < 0) | (acts["credit"].fillna(0) < 0)]
    if len(neg_money):
        qlog.add("odd_money_sign", "acts.negative", len(neg_money),
                 str(neg_money["doc"].iloc[0]), dedup=True)
    double_sided = acts[(acts["debet"].fillna(0) != 0) & (acts["credit"].fillna(0) != 0)]
    if len(double_sided):
        qlog.add("odd_money_sign", "acts.debet_and_credit", len(double_sided),
                 str(double_sided["doc"].iloc[0]), dedup=True)
    zero_rows = acts[(acts["debet"].fillna(0) == 0) & (acts["credit"].fillna(0) == 0)]
    if len(zero_rows):
        qlog.add("odd_money_sign", "acts.zero_sum", len(zero_rows),
                 str(zero_rows["doc"].iloc[0]), dedup=True)
    acts["ticket_set"] = acts["ticket_cell"].map(lambda v: ticket_numbers(v, cfg))

    short_min, short_tail = cfg["tickets"]["min_digits"], cfg["tickets"]["tail"]
    short = acts.loc[acts["ticket_cell"].notna(), "ticket_cell"].astype(str)
    short = short[~short.str.contains(r"\d{" + str(short_min) + r",}")]
    if len(short):
        qlog.add("short_ticket_number", "acts.ticket_cell", len(short), short.iloc[0], dedup=True)
    short_e = etm.loc[etm["tickets"].notna(), "tickets"].astype(str)
    short_e = short_e[~short_e.str.contains(r"\d{" + str(short_min) + r",}")]
    if len(short_e):
        qlog.add("short_ticket_number", "etm.tickets", len(short_e), short_e.iloc[0], dedup=True)

    etm["sum_etm"] = resolve_sum_etm(etm, reg, cfg, qlog)
    etm["ticket_set"] = etm["tickets"].map(lambda v: ticket_numbers(v, cfg))
    reg["ticket_set"] = reg["tickets"].map(lambda v: ticket_numbers(v, cfg))

    # --- sloy imen: reestr skлеиvaetsya s yadrom, yadro (akty+ETM) ne trogaem ---
    # Dengi khodyat tolko po exact+alias (sm. vyshe); zdes razreshaem storonu
    # reestra: exact -> nospace -> homoglyph -> strogiy fuzzy -> orphan.
    core_list = sorted(set(acts["agent_norm"]) | set(etm["agent_norm"]))
    name_index = build_name_index(core_list, cfg)
    tickets_of = agent_ticket_sets(acts, etm)
    reg["_party_norm"] = reg["party"].astype(str).map(
        lambda v: apply_aliases(norm_agent(v, cfg), cfg))
    party_tickets = reg.groupby("_party_norm")["ticket_set"].agg(
        lambda sets: set().union(*[s for s in sets.tolist()]))
    party_counts = reg["_party_norm"].value_counts()
    party_labels = reg.groupby("_party_norm")["party"].first()
    resolution: dict[str, str] = {}
    for pnorm in party_tickets.index:
        resolution[pnorm] = resolve_registry_party(
            pnorm, set(party_tickets.loc[pnorm]), name_index, core_list,
            tickets_of, cfg, qlog, str(party_labels.loc[pnorm]),
            int(party_counts.loc[pnorm]))
    reg["agent_norm"] = reg["_party_norm"].map(lambda p: resolution.get(p, p))
    reg = reg.drop(columns=["_party_norm"])

    # Doc ne togo tipa: stroka s biletami, no doc platezhnyy (i naoborot).
    # Izvestnye tipy: bilety — Реализация/Возврат; platezhi — nomer ЦБ-С/platezhnye slova.
    pay_like = acts["doc"].astype(str).str.contains(
        r"ЦБ-С|Поступление|Приход|касс|оплат", na=False)
    ticket_like = acts["doc"].astype(str).str.contains(r"Реализация|Возврат", na=False)
    odd_tickets = acts.loc[acts["ticket_cell"].notna() & pay_like]
    if len(odd_tickets):
        qlog.add("doc_ticket_mismatch", "ticket_row_with_payment_doc",
                 len(odd_tickets), str(odd_tickets["doc"].iloc[0]), dedup=True)
    odd_payments = acts.loc[acts["ticket_cell"].isna() & ~pay_like & ticket_like]
    if len(odd_payments):
        qlog.add("doc_ticket_mismatch", "payment_row_with_ticket_doc",
                 len(odd_payments), str(odd_payments["doc"].iloc[0]), dedup=True)

    dup_acts = int(acts.duplicated(
        subset=["folder", "period_start", "date", "doc", "ticket_cell"]).sum())
    if dup_acts:
        qlog.add("duplicate_rows", "acts", dup_acts, "", dedup=True)
    if "txn_id" in etm.columns:
        dup_etm = int(etm.duplicated(subset=["txn_id"]).sum())
        if dup_etm:
            qlog.add("duplicate_rows", "etm.txn_id", dup_etm, "", dedup=True)
    return acts, etm, reg, dict(ignored_drafts)


# ------------------------------------------------------------ components ---

def canonical_labels(acts: pd.DataFrame, etm: pd.DataFrame, cfg: dict) -> dict:
    """Edinaya podpis agenta v otchetakh.
    acts_first (= povedenie v1: ETM-nazvaniya chasto v drugom registre/prefixe,
    poetomu berem podpis iz 1C, fallback na ETM) libo most_frequent."""
    source = (cfg.get("labels") or {}).get("source", "acts_first")
    if source == "most_frequent" and len(acts) + len(etm):
        combo = pd.concat([acts[["agent_norm", "agent_label"]],
                           etm[["agent_norm", "agent_label"]]])
        return combo.groupby("agent_norm")["agent_label"].agg(
            lambda s: s.value_counts().idxmax()).to_dict()
    return {**etm.groupby("agent_norm")["agent_label"].first().to_dict(),
            **acts.groupby("agent_norm")["agent_label"].first().to_dict()}


def ticket_components(acts: pd.DataFrame, etm: pd.DataFrame) -> list[tuple[list, list]]:
    """Svyaznye komponenty strok 1C i ETM cherez obshchie bilety (many-to-many)."""
    nodes = [("a", i) for i in acts.index] + [("e", i) for i in etm.index]
    parent = {node: node for node in nodes}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        lr, rr = find(left), find(right)
        if lr != rr:
            parent[rr] = lr

    by_ticket: dict[str, list] = defaultdict(list)
    for i, tickets in acts["ticket_set"].items():
        for ticket in tickets:
            by_ticket[ticket].append(("a", i))
    for i, tickets in etm["ticket_set"].items():
        for ticket in tickets:
            by_ticket[ticket].append(("e", i))
    for ticket_nodes in by_ticket.values():
        for node in ticket_nodes[1:]:
            union(ticket_nodes[0], node)

    components: dict = defaultdict(list)
    for node in nodes:
        components[find(node)].append(node)
    return [
        ([i for source, i in comp if source == "a"],
         [i for source, i in comp if source == "e"])
        for comp in components.values()
    ]
