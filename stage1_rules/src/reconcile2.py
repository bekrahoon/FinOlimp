"""Sverka 1C i ETM dlya stage1_rules (universalnyy rule-based dvizhok).

Chitaet acts.csv / etm.csv / registry.csv i stroit:
- discrepancies.csv (agent,date,identifier,sum_1c,sum_etm,diff,comment,source)
- reconciliation_summary.csv (svyazka saldo 1C i balansa ETM)
- review_candidates.csv (typed, non-authoritative manual-audit queue)
- quality_report.csv (inventar neizvestnogo: novye znacheniya, dlinnye gapy,
  kross-agentnye bilety, problemy kodirovok/daty/valyut)

Vse pravila vyneseny v reconcile.yaml ryadom so skriptom:
dopuski, stop-slova imen, politika statusov, okno platezhey, valyuty/kursy,
aliasy agentov (pereimenovaniya), diagnoz voyda s ostatkom.
Bez konfiga rabotayut vstroennye defolty = povedenie pervoy versii.

Otlichiya ot pervoy versii (thed zhe columns, lyubye dannye):
- avtodetekt kodirovki (utf-8-sig/utf-8/cp1251), strip strokovykh kletok;
- avtoparsing dat po spisku formatov + unparseable_date v otchet;
- chistka denezhnykh formatov ('1 234,56', $/rub, skobki-otritsatelnye);
- platezhi: snachala tochnyy match po nomeru ЦБ-С (kolonka comment v ETM),
  zatem greedy po summe vnutri date_window_days (defolt 120);
  matchi s gap > review_gap_days idut v quality_report na proverku;
- valyuty: esli net gotovoy amount_kgs — pereschet iz amount*kurs
  (kursy iz reestra na datu <= date operatsii, libo fallback_rates);
- pereimenovaniya agentov: agents.aliases + kross-agentnyy fallback
  (bilet, ne nashedshiy paru v svoem agente, ishchetsya globalno);
- summary ne padaet na odnostoronnikh agentakh (one_sided=True, NaN);
- voyd s ostatkom: otdelnyy diagnoz za flagom comments.void_remainder_note
  (po umolchaniyu vyklyuchen — sokhranyaet starye kommentarii).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Obshiy parsing/zagruzka: sm. parse.py (edinaya tochka pravok dlya
# dvizhka i matchera; posle pravok — golden-sverka + pytest + matcher-diff).
from parse import (
    HERE,
    DEFAULT_CONFIG,
    QualityLog,
    apply_aliases,
    canonical_labels,
    doc_ref,
    load_config,
    load_data,
    norm_agent,
    payment_identifier,
    parse_pay_kgs_total,
    resolve_data_end,
    ticket_components,
    ticket_numbers,
)
from review_candidates import build_review_candidates

# DEFAULT_CONFIG, REQUIRED_* — sm. parse.py (edinyy istochnik).


# config/load/parse/names/currency/ticket-keys — sm. parse.py.

# (prodolzhenie perenosa — sm. parse.py)


# (prodolzhenie perenosa — sm. parse.py)


# (prodolzhenie perenosa — sm. parse.py)


# ------------------------------------------------------------ components ---

# canonical_labels — sm. parse.py.


def check_multi_label_agents(acts: pd.DataFrame, etm: pd.DataFrame,
                             qlog: QualityLog) -> None:
    """Raznye syrye podpisi vnutri odnoy storony (pereimenovanie v seredine fayla):
    operation_diff mozhet molcha ne sostykovatsya po metke."""
    for name, frame in (("acts", acts), ("etm", etm)):
        if not len(frame):
            continue
        nunique = frame.groupby("agent_norm")["agent_label"].apply(
            lambda s: len(set(s.astype(str).str.strip())))
        for agent, count in nunique[nunique > 1].items():
            labels = sorted(set(frame.loc[frame["agent_norm"] == agent, "agent_label"]
                                .astype(str).str.strip()))
            qlog.add("multi_label_agent", f"{name}:{agent}", int(count),
                     " | ".join(labels), dedup=True)


# ticket_components — sm. parse.py.


def global_ticket_agents(acts: pd.DataFrame, etm: pd.DataFrame) -> dict[str, set[str]]:
    """Globalnyy indeks: bilet -> mnozhestvo agent_norm (dlya kross-agent fallback)."""
    index: dict[str, set[str]] = defaultdict(set)
    for _, row in acts.iterrows():
        for ticket in row["ticket_set"]:
            index[ticket].add(row["agent_norm"])
    for _, row in etm.iterrows():
        for ticket in row["ticket_set"]:
            index[ticket].add(row["agent_norm"])
    return index


def registry_lookup(reg: pd.DataFrame) -> dict:
    """Indeks reestra: ticket -> mnozhestvo employee (tolko nenull ticket_set)."""
    by_ticket_emp: dict[str, set[str]] = defaultdict(set)
    for _, row in reg.iterrows():
        emp = str(row.get("employee", "") or "").strip()
        for ticket in row["ticket_set"]:
            if emp:
                by_ticket_emp[ticket].add(emp)
    return by_ticket_emp


# -------------------------------------------------------------- comments ---

def has_void(etm_rows: pd.DataFrame, cfg: dict) -> bool:
    if "kind" not in etm_rows.columns or etm_rows.empty:
        return False
    kinds = {k.lower() for k in cfg.get("void_kinds", [])}
    return bool(etm_rows["kind"].astype(str).str.strip().str.lower().isin(kinds).any())


def discrepancy_comment(
    sum_1c: float,
    sum_etm: float,
    act_rows: pd.DataFrame,
    etm_rows: pd.DataFrame,
    in_registry: bool,
    cfg: dict | None = None,
    void_present: bool = False,
) -> str:
    tol = (cfg or DEFAULT_CONFIG)["tolerance"]
    reg_note = "; в реестре есть" if in_registry else "; в реестре нет"
    if act_rows.empty:
        if not etm_rows.empty and (etm_rows["creator"].astype(str) == "субагент").all() \
                if "creator" in etm_rows else False:
            return ("Есть только в ETM (самовыписка субагента); "
                    "вероятно, не отражено в 1С" + reg_note)
        if not etm_rows.empty and (etm_rows["creator"].astype(str) == "etm-bot").all() \
                if "creator" in etm_rows else False:
            if in_registry:
                return ("Есть только в ETM (бот); "
                        "вероятно, операция не отражена в 1С" + reg_note)
            return ("Есть в ETM (бот), но нет в 1С и в реестре; "
                    "проверить запись агента" + reg_note)
        return "Есть только в ETM; вероятно, операция не отражена в 1С" + reg_note
    if etm_rows.empty:
        if in_registry:
            return ("Есть в 1С и реестре, нет в ETM; "
                    "вероятно, бот пропустил строку" + reg_note)
        return "Есть только в 1С; вероятно, операция не проведена в ETM" + reg_note
    etm_ticket_kinds = [
        (ticket, row["kind"])
        for _, row in etm_rows.iterrows()
        for ticket in row["ticket_set"]
    ]
    if len(etm_ticket_kinds) != len(set(etm_ticket_kinds)):
        return "Суммы отличаются; в ETM вероятен дубль операции" + reg_note
    if abs(sum_etm) < tol:
        return "Операция аннулирована в ETM, но осталась в 1С; проверить 1С" + reg_note
    if void_present and (cfg or DEFAULT_CONFIG)["comments"]["void_remainder_note"]:
        return (f"Войд в ETM закрыт частично, остаток {sum_1c - sum_etm:.2f}; "
                f"проверить комиссию/пересчет" + reg_note)
    return "Суммы операции отличаются; требуется проверка первичных документов" + reg_note


# --------------------------------------- source (Шаг 4: бот/агент/1С/выяснить) ---
# cited sources: легенда FinOlimp (бот переносит реестр->ETM, агент ведет реестр
# вручную, 1С ведет бухгалтерия). Маппинг по префиксу комментария (суффиксы
# "(реестр: ...)" и "(v3 ...)" не влияют). Класс "агент" маппингом не покрывается:
# ошибка агента в реестре проявляется как sum_diff с реестром, где сумма реестра
# == сумме ETM — см. v3 R1 ниже (конверсия Q2 + adjudication 150).
#
# v3 mikroruly (Q1; cfg["v3"], enabled=False = rezhim v2):
# - R1: sum_diff + summa reestra ~= 1C (konversiya, conv_tol) + tolko bot-sozdateli
#   + |diff| > r1_min_diff -> bot (ETM-zanizheniya S111/S114/S115/S120/S123/S124).
#   Osoznannoe isklyuchenie: S112 (rovno 1000.00 — ruchnaya pravka, storona agent).
# - R2 (reg ~= ETM): chastichno avtomatizirovan kak RA nizhe; ostatok (smeshannye
#   sluchai) — bez izmeneniy, chestno ne delim.
# - RA (agent-klass): sum_diff + summa reestra ~= ETM (konversiya, conv_tol) +
#   tolko bot-sozdateli + ODNA stroka reestra + diff > ra_min_diff + diff > 0 +
#   ne svezhaya -> agent (S110/S117/S119/S122: bot skopiroval dobrosovestno,
#   oshibka — v uchete agenta). S100-shtraf otsekaetsya odnostrochnostyu.
# - R3a/R3b: ETM-only staryy platyoz: est bliznets (ta zhe summa+vid, <=7d) -> bot
#   (dubl: 22 pary s doc-nogoy v 1С + S131); net bliznetsa -> 1С (S139/S143/S148).
# - R4/R5: bez izmeneniy (shtraf/storno; singltony S106/S141 ne kodifitsiruem).

SOURCE_BY_PREFIX = (
    ("Есть в 1С и реестре, нет в ETM", "бот"),          # бот пропустил строку
    ("Есть в ETM (бот), но нет в 1С и в реестре", "бот"),  # фантомная запись
    ("Суммы отличаются", "бот"),                        # дубль в ETM
    ("Есть только в ETM (бот)", "1С"),                  # операция реальна (реестр), 1С молчит
    ("Есть только в ETM (самовыписка субагента)", "1С"),  # самовыписка не отражена в 1С
    ("Есть только в ETM", "1С"),                        # fallback без creator
    ("Операция аннулирована в ETM, но осталась в 1С", "1С"),
    ("Платеж есть только в 1С", "выяснить"),            # timing vs невнесение в ETM
    ("Платеж есть только в ETM", "выяснить"),           # невнесение в 1С vs дубль
    ("Суммы операции отличаются", "выяснить"),          # комиссии vs ошибка ввода
    ("Есть только в 1С", "выяснить"),
    ("Войд в ETM закрыт частично", "выяснить"),
)


def source_for_comment(comment: str, qlog=None) -> str:
    text = str(comment)
    for prefix, source in SOURCE_BY_PREFIX:
        if text.startswith(prefix):
            return source
    if qlog is not None:  # never-fail: наблюдаем, не падаем
        qlog.add("unknown_source_prefix", text[:60], 1, "", dedup=True)
    return "выяснить"


# ------------------------------------------------- v3 mikroruly (Q1) ---

def v3_cfg(cfg: dict) -> dict:
    """Sektsiya cfg['v3'] ili {} (enabled=False = rezhim v2). Never-fail."""
    try:
        v3 = (cfg or {}).get("v3", {}) or {}
    except AttributeError:
        return {}
    if not v3.get("enabled", True):
        return {}
    return v3


def v3_data_end(v3: dict, qlog=None, acts=None, etm=None):
    """Data vygruzki: delegiruem P.resolve_data_end ('auto' = max dat)."""
    return resolve_data_end({"v3": v3 or {}}, acts, etm, qlog)


def ensure_pay_conv(reg: pd.DataFrame) -> pd.DataFrame:
    """Konvertirovannye summy reestra (Q2); schitaem lenivo, esli kolonki net."""
    if "pay_conv" in reg.columns:
        return reg
    reg = reg.copy()
    reg["pay_conv"] = reg.apply(
        lambda x: parse_pay_kgs_total(x.get("pay_cell"), x), axis=1)
    return reg


def v3_r1_source(sum_1c: float, sum_etm: float, diff: float, tickets: list,
                 agent: str, etm_rows: pd.DataFrame, reg: pd.DataFrame,
                 reg_by_ticket: dict, v3: dict, row_date, fresh_cut):
    """R1: sum_diff + reestr ~= 1C + tolko bot -> 'бот'. Inache None.
    Gardy (spot-check 25, Q1): tolko ODNA stroka reestra (mnogo-strochnye
    baza+shtraf-arithmetiki = klass R4: Baytur/Karakol-27000/Mambetova/Cholpon)
    i tolko starye stroki (timing-schit: svezhie mogut doekhat v ETM pozzhe)."""
    if abs(diff) <= float(v3.get("r1_min_diff", 500.0)):
        return None
    if fresh_cut is not None and pd.notna(row_date) and row_date > fresh_cut:
        return None
    creators = (set(etm_rows["creator"].astype(str))
                if "creator" in etm_rows.columns and len(etm_rows) else set())
    if creators != {"etm-bot"}:
        return None
    idx = sorted({i for t in tickets for i in reg_by_ticket.get(t, [])
                  if reg.loc[i, "agent_norm"] == agent})
    if len(idx) != 1 or pd.isna(reg.loc[idx[0], "pay_conv"]):
        return None
    rs = round(float(reg.loc[idx[0], "pay_conv"]), 2)
    tol = float(v3.get("conv_tol", 2.0))
    if abs(rs - abs(sum_1c)) < tol and abs(rs - abs(sum_etm)) >= tol:
        return ("бот", " (v3 R1: сумма реестра ≈ 1С)",
                f"{rs} ≈ 1С {sum_1c} (рег.строка {reg.loc[idx[0], 'date']})")
    return None


def v3_rA_source(sum_1c: float, sum_etm: float, diff: float, tickets: list,
                 agent: str, etm_rows: pd.DataFrame, reg: pd.DataFrame,
                 reg_by_ticket: dict, v3: dict, row_date, fresh_cut):
    """RA (agent-klass): sum_diff + reestr ~= ETM + tolko bot -> 'агент'. Inache None.
    Yakorya: S110/S117/S119/S122 (reestr==ETM!=1С: bot skopiroval dobrosovestno,
    oshibka — v uchete agenta). Gardy (spot-check, Q1-analog R1): tolko ODNA
    stroka reestra (S100 shtraf: baza+shtraf = klass R4), tolko polozhitelnyy
    diff (ETM zanizhen; genuine S102 otritsatelen), |diff| > ra_min_diff
    (genuine-meloch), ne svezhaya (timing-schit). S112/S116 — ne scope RA
    (u nih reestr==1С, eto R1-miss)."
    """
    if abs(diff) <= float(v3.get("ra_min_diff", 500.0)):
        return None
    if diff <= 0:
        return None
    if fresh_cut is not None and pd.notna(row_date) and row_date > fresh_cut:
        return None
    creators = (set(etm_rows["creator"].astype(str))
                if "creator" in etm_rows.columns and len(etm_rows) else set())
    if creators != {"etm-bot"}:
        return None
    idx = sorted({i for t in tickets for i in reg_by_ticket.get(t, [])
                  if reg.loc[i, "agent_norm"] == agent})
    if len(idx) != 1 or pd.isna(reg.loc[idx[0], "pay_conv"]):
        return None
    rs = round(float(reg.loc[idx[0], "pay_conv"]), 2)
    tol = float(v3.get("conv_tol", 2.0))
    if abs(rs - abs(sum_etm)) < tol and abs(rs - abs(sum_1c)) >= tol:
        return ("агент", " (v3 RA: сумма реестра ≈ ETM)",
                f"{rs} ≈ ETM {sum_etm} (рег.строка {reg.loc[idx[0], 'date']})")
    return None


def v3_r3_source(txn, etm_id, etm_rows: pd.DataFrame, tol: float,
                 twin_window: pd.Timedelta):
    """R3: ETM-only platyoz s bliznetsom -> 'бот' (dubl), bez -> '1С'."""
    sib = etm_rows[
        (etm_rows.index != etm_id)
        & (etm_rows["kind"].astype(str) == str(txn["kind"])
           if "kind" in etm_rows.columns else True)
        & ((etm_rows["sum_etm"] - float(txn["sum_etm"])).abs() < tol)
        & ((etm_rows["date_norm"] - txn["date_norm"]).abs() <= twin_window)
    ] if pd.notna(txn["date_norm"]) else etm_rows.iloc[0:0]
    if len(sib):
        mate = str(sib.iloc[0]["txn_id"])
        return ("бот", " (v3 R3b: в ETM есть близнец, вероятно дубль)",
                f"близнец txn {mate}")
    return ("1С", " (v3 R3a: близнецов в ETM нет, 1С молчит)", "одиночный")


# --------------------------------------------------------------- tickets ---

def reconcile_tickets(
    acts: pd.DataFrame,
    etm: pd.DataFrame,
    reg: pd.DataFrame,
    cfg: dict,
    qlog: QualityLog,
) -> list[dict[str, object]]:
    tol = cfg["tolerance"]
    by_ticket_emp = registry_lookup(reg)
    ticket_agents = global_ticket_agents(acts, etm)
    reg = ensure_pay_conv(reg)
    reg_by_ticket: dict = defaultdict(list)
    for i, ts in reg["ticket_set"].items():
        for t in (ts if isinstance(ts, set) else set()):
            reg_by_ticket[t].append(i)
    v3 = v3_cfg(cfg)
    data_end = v3_data_end(v3, qlog, acts, etm) if v3 else None
    fresh_cut = (data_end - pd.Timedelta(days=int(v3.get("fresh_days", 31)))) \
        if data_end is not None else None
    rows: list[dict[str, object]] = []
    agents = sorted(set(acts["agent_norm"]) | set(etm["agent_norm"]))
    labels = canonical_labels(acts, etm, cfg)
    for agent in agents:
        acts_agent = acts[(acts["agent_norm"] == agent) & acts["ticket_cell"].notna()]
        etm_agent = etm[(etm["agent_norm"] == agent) & etm["tickets"].notna()]
        reg_tickets = set()
        for tickets in reg.loc[reg["agent_norm"] == agent, "ticket_set"]:
            reg_tickets |= set(tickets)
        for act_ids, etm_ids in ticket_components(acts_agent, etm_agent):
            act_rows, etm_rows = acts.loc[act_ids], etm.loc[etm_ids]
            sum_1c = round(float(act_rows["sum_1c"].sum()), 2)
            sum_etm = round(float(etm_rows["sum_etm"].sum()), 2)
            diff = round(sum_1c - sum_etm, 2)
            if abs(diff) < tol:
                continue
            all_tickets = sorted(
                set().union(*act_rows["ticket_set"].tolist(), *etm_rows["ticket_set"].tolist())
            )
            in_registry = bool(set(all_tickets) & reg_tickets)
            employees = sorted({e for t in all_tickets for e in by_ticket_emp.get(t, set())})
            comment = discrepancy_comment(sum_1c, sum_etm, act_rows, etm_rows,
                                          in_registry, cfg, has_void(etm_rows, cfg))
            if employees:
                comment += f" (реестр: {', '.join(employees)})"
            # Kross-agentnyy fallback: odnostoronnyaya komponenta, no bilet est
            # u drugogo agenta — vozmozhnoe pereimenovanie.
            if not act_rows.empty and etm_rows.empty or act_rows.empty and not etm_rows.empty:
                others = sorted({a for t in all_tickets for a in ticket_agents.get(t, set())}
                                - {agent})
                if others:
                    other_labels = ", ".join(labels.get(o, o) for o in others)
                    comment += f" (кросс-агент: {other_labels}; возможно переименование)"
                    qlog.add("cross_agent_ticket", agent, 1, other_labels, dedup=True)
            dates = pd.concat([act_rows["date_norm"], etm_rows["date_norm"]])
            source = source_for_comment(comment, qlog)
            if v3 and fresh_cut is not None \
                    and comment.startswith("Суммы операции отличаются"):
                r1 = v3_r1_source(sum_1c, sum_etm, diff, all_tickets, agent,
                                  etm_rows, reg, reg_by_ticket, v3,
                                  dates.min(), fresh_cut)
                if r1:
                    source, suffix, note = r1
                    comment += suffix
                    qlog.add("v3_r1", agent, 1, note, dedup=True)
                else:
                    rA = v3_rA_source(sum_1c, sum_etm, diff, all_tickets, agent,
                                      etm_rows, reg, reg_by_ticket, v3,
                                      dates.min(), fresh_cut)
                    if rA:
                        source, suffix, note = rA
                        comment += suffix
                        qlog.add("v3_rA", agent, 1, note, dedup=True)
            rows.append({
                "agent": labels.get(agent, agent),
                "date": dates.min().date().isoformat() if dates.notna().any() else "",
                "identifier": "|".join(all_tickets),
                "sum_1c": sum_1c,
                "sum_etm": sum_etm,
                "diff": diff,
                "comment": comment,
                "source": source,
            })
    return rows


# -------------------------------------------------------------- payments ---

def reconcile_payments(acts: pd.DataFrame, etm: pd.DataFrame, cfg: dict,
                       qlog: QualityLog) -> list[dict[str, object]]:
    tol = cfg["tolerance"]
    pcfg = cfg["payments"]
    window = pd.Timedelta(days=pcfg["date_window_days"])
    v3 = v3_cfg(cfg)
    data_end = v3_data_end(v3, qlog, acts, etm) if v3 else None
    fresh_cut = (data_end - pd.Timedelta(days=int(v3.get("fresh_days", 31)))) \
        if data_end is not None else None
    twin_window = pd.Timedelta(days=int(v3.get("twin_window_days", 7)) if v3 else 7)
    rows: list[dict[str, object]] = []
    agents = sorted(set(acts["agent_norm"]) | set(etm["agent_norm"]))
    labels = canonical_labels(acts, etm, cfg)
    etm_ref_col = "comment" if "comment" in etm.columns else None
    for agent in agents:
        act_rows = acts[(acts["agent_norm"] == agent) & acts["ticket_cell"].isna()].copy()
        etm_rows = etm[(etm["agent_norm"] == agent) & etm["tickets"].isna()].copy()
        act_rows["_ref"] = act_rows["doc"].map(lambda d: doc_ref(d, cfg))
        if etm_ref_col:
            etm_rows["_ref"] = etm_rows[etm_ref_col].map(lambda c: doc_ref(c, cfg))
        else:
            etm_rows["_ref"] = ""
        unmatched_etm = set(etm_rows.index)
        unmatched_acts: list = []
        act_rows = act_rows.sort_values("date_norm")

        def take_nearest(candidates: set, when) -> object:
            return min(candidates,
                       key=lambda i: (abs((etm_rows.loc[i, "date_norm"] - when).total_seconds()), i))

        # Faza 1 (tochnaya): sovpadenie summy + nomera ЦБ-С — vne zavisimosti ot okna.
        act_rows["_matched"] = False
        if pcfg["prefer_doc_match"]:
            by_ref: dict[tuple, list] = defaultdict(list)
            for i in sorted(unmatched_etm):
                if etm_rows.loc[i, "_ref"]:
                    by_ref[(round(float(etm_rows.loc[i, "sum_etm"]), 2),
                            etm_rows.loc[i, "_ref"])].append(i)
            for act_id, act in act_rows.iterrows():
                if act["_ref"]:
                    pool = by_ref.get((round(float(act["sum_1c"]), 2), act["_ref"]), [])
                    pool = [i for i in pool if i in unmatched_etm]
                    if pool:
                        chosen = min(pool, key=lambda i: (
                            abs((etm_rows.loc[i, "date_norm"] - act["date_norm"]).total_seconds()), i))
                        unmatched_etm.remove(chosen)
                        act_rows.loc[act_id, "_matched"] = True
                        if len(pool) > 1:
                            rej = ", ".join(str(etm_rows.loc[i, "txn_id"]) for i in pool
                                            if i != chosen)
                            qlog.add("payment_tiebreak", agent, 1,
                                     f"{act['doc']} -> {etm_rows.loc[chosen, 'txn_id']} "
                                     f"(tochnyy doc, otbrosheno: {rej})")
            act_rows["_matched"] = act_rows["_matched"].fillna(False).astype(bool)
        for act_id, act in act_rows.loc[~act_rows["_matched"]].sort_values("date_norm").iterrows():
            if pd.isna(act["date_norm"]):
                unmatched_acts.append(act_id)
                continue
            candidates = {
                i for i in unmatched_etm
                if abs(float(etm_rows.loc[i, "sum_etm"]) - float(act["sum_1c"])) < tol
                and not pd.isna(etm_rows.loc[i, "date_norm"])
                and abs(etm_rows.loc[i, "date_norm"] - act["date_norm"]) <= window
            }
            if candidates:
                chosen = take_nearest(candidates, act["date_norm"])
                unmatched_etm.remove(chosen)
                gap_days = abs((etm_rows.loc[chosen, "date_norm"]
                                - act["date_norm"]).total_seconds()) / 86400
                if gap_days > pcfg["review_gap_days"]:
                    qlog.add("long_gap_payment", agent, 1,
                             f"{act['doc']} <-> {etm_rows.loc[chosen, 'txn_id']} ({gap_days:.0f} дн.)")
                if len(candidates) > 1:
                    dists = sorted(
                        (abs((etm_rows.loc[i, "date_norm"] - act["date_norm"]).total_seconds()), i)
                        for i in candidates)
                    rej = ", ".join(f"{etm_rows.loc[i, 'txn_id']}@{d / 86400:.1f}дн"
                                    for d, i in dists[1:4])
                    qlog.add("payment_tiebreak", agent, 1,
                             f"{act['doc']} -> {etm_rows.loc[chosen, 'txn_id']}; "
                             f"otbrosheno: {rej}")
            else:
                unmatched_acts.append(act_id)
        for act_id in unmatched_acts:
            act = acts.loc[act_id]
            rows.append({
                "agent": labels.get(agent, agent),
                "date": act["date_norm"].date().isoformat() if pd.notna(act["date_norm"]) else "",
                "identifier": payment_identifier(str(act["doc"])),
                "sum_1c": float(act["sum_1c"]),
                "sum_etm": 0.0,
                "diff": float(act["sum_1c"]),
                "comment": "Платеж есть только в 1С; вероятно, не зачислен в ETM",
                "source": source_for_comment("Платеж есть только в 1С", qlog),
            })
        for etm_id in unmatched_etm:
            txn = etm.loc[etm_id]
            source = source_for_comment("Платеж есть только в ETM", qlog)
            suffix = ""
            if v3 and fresh_cut is not None and pd.notna(txn["date_norm"]) \
                    and txn["date_norm"] <= fresh_cut:
                source, suffix, note = v3_r3_source(txn, etm_id, etm_rows, tol,
                                                   twin_window)
                qlog.add("v3_r3a_r3b", agent, 1,
                         f"{txn['txn_id']}: {source} ({note})", dedup=True)
            rows.append({
                "agent": labels.get(agent, agent),
                "date": txn["date_norm"].date().isoformat() if pd.notna(txn["date_norm"]) else "",
                "identifier": str(txn["txn_id"]),
                "sum_1c": 0.0,
                "sum_etm": float(txn["sum_etm"]),
                "diff": round(-float(txn["sum_etm"]), 2),
                "comment": "Платеж есть только в ETM; вероятно, не отражен в 1С или задублирован" + suffix,
                "source": source,
            })
    return rows


# --------------------------------------------------------------- summary ---

def opening_etm_balance(etm_rows: pd.DataFrame) -> float:
    """Nachalnyy balans ETM iz pervoy operatsii (rabotaet i bez amount_kgs)."""
    if not len(etm_rows):
        return float("nan")
    first = etm_rows.iloc[0]
    if "amount_kgs" in etm_rows.columns:
        return float(first["balance_after"] - first["amount_kgs"])
    return float(first["balance_after"] + first["sum_etm"])


def reconciliation_summary(
    acts: pd.DataFrame,
    etm: pd.DataFrame,
    discrepancies: pd.DataFrame,
    ignored_drafts: dict[str, int],
    cfg: dict | None = None,
) -> pd.DataFrame:
    tol = (cfg or DEFAULT_CONFIG)["tolerance"]
    result: list[dict[str, object]] = []
    labels = canonical_labels(acts, etm, cfg or DEFAULT_CONFIG)
    for agent in sorted(set(acts["agent_norm"]) | set(etm["agent_norm"])):
        act_rows = acts[acts["agent_norm"] == agent].sort_values(["period_start_norm", "date_norm"])
        etm_rows = etm[etm["agent_norm"] == agent].sort_values("date_norm")
        label = labels.get(agent, agent)
        one_sided = not len(act_rows) or not len(etm_rows)
        aliased = sorted(set(
            [s for s in act_rows["_aliased_from"].tolist() if s]
            + [s for s in etm_rows["_aliased_from"].tolist() if s]
        )) if "_aliased_from" in act_rows.columns else []
        if one_sided:
            result.append({
                "agent": label,
                "opening_saldo_1c": float("nan") if not len(act_rows) else float(
                    act_rows.groupby("period_start_norm", sort=True)["saldo_start"].first().iloc[0]),
                "opening_balance_etm": opening_etm_balance(etm_rows),
                "opening_gap": float("nan"),
                "interperiod_1c_gap": float("nan"),
                "operation_diff": round(float(discrepancies.loc[discrepancies.agent == label, "diff"].sum()), 2),
                "ending_saldo_1c": float("nan"),
                "ending_balance_etm": float("nan"),
                "ending_gap": float("nan"),
                "explained_gap": float("nan"),
                "residual": float("nan"),
                "operations_explain_movement": False,
                "ignored_draft_rows": ignored_drafts.get(agent, 0),
                "one_sided": True,
                "alias_of": "; ".join(aliased),
            })
            continue
        periods = act_rows.groupby("period_start_norm", sort=True)
        starts = periods["saldo_start"].first()
        ends = periods["saldo_end"].first()
        period_gap = round(float(sum(starts.iloc[1:].values - ends.iloc[:-1].values)), 2)
        opening_1c = float(starts.iloc[0])
        opening_etm = opening_etm_balance(etm_rows)
        ending_1c = float(ends.iloc[-1])
        ending_etm = float(etm_rows.iloc[-1]["balance_after"])
        opening_gap = round(opening_1c + opening_etm, 2)
        operation_diff = round(float(discrepancies.loc[discrepancies.agent == label, "diff"].sum()), 2)
        ending_gap = round(ending_1c + ending_etm, 2)
        explained_gap = round(opening_gap + period_gap + operation_diff, 2)
        residual = round(ending_gap - explained_gap, 2)
        result.append({
            "agent": label,
            "opening_saldo_1c": opening_1c,
            "opening_balance_etm": opening_etm,
            "opening_gap": opening_gap,
            "interperiod_1c_gap": period_gap,
            "operation_diff": operation_diff,
            "ending_saldo_1c": ending_1c,
            "ending_balance_etm": ending_etm,
            "ending_gap": ending_gap,
            "explained_gap": explained_gap,
            "residual": residual,
            "operations_explain_movement": abs(residual) < tol,
            "ignored_draft_rows": ignored_drafts.get(agent, 0),
            "one_sided": False,
            "alias_of": "; ".join(aliased),
        })
    return pd.DataFrame(result)


def check_date_ranges(acts: pd.DataFrame, etm: pd.DataFrame, cfg: dict,
                      qlog: QualityLog) -> None:
    """Daty vne okna periodov (+margin) — veroyatnye opechatki goda (2025/2036)."""
    if not len(acts) or acts["period_start_norm"].isna().all() \
            or acts["period_end_norm"].isna().all():
        return
    margin = pd.Timedelta(days=cfg.get("dates", {}).get("range_margin_days", 15))
    lo = acts["period_start_norm"].min() - margin
    hi = acts["period_end_norm"].max() + margin
    for name, frame in (("acts", acts), ("etm", etm)):
        valid = frame.loc[frame["date_norm"].notna()
                          & ((frame["date_norm"] < lo) | (frame["date_norm"] > hi))]
        for agent, group in valid.groupby(frame["agent_norm"] if "agent_norm" in frame.columns else []):
            qlog.add("date_out_of_range", f"{name}:{agent}", len(group),
                     str(group["date_norm"].iloc[0].date()), dedup=True)


def check_missing_periods(acts: pd.DataFrame, etm: pd.DataFrame, cfg: dict,
                          qlog: QualityLog) -> None:
    """Vnutrennie дыры mesyatsev: ETM aktivna v dyre — poteryan fayl aktov (krasnyy),
    inache — tikhiy prostoy agenta."""
    if not len(acts) or acts["period_start_norm"].isna().all():
        return
    grid = pd.period_range(acts["period_start_norm"].min(),
                           acts["period_start_norm"].max(), freq="M").astype(str).tolist()

    def months_of(frame: pd.DataFrame) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        valid = frame.loc[frame["date_norm"].notna()]
        for agent, group in valid.groupby("agent_norm"):
            result[agent] = set(group["date_norm"].dt.to_period("M").astype(str))
        return result

    acts_months = months_of(acts)
    etm_months = months_of(etm)
    for agent in sorted(set(acts_months) | set(etm_months)):
        for month in grid:
            if month not in acts_months.get(agent, set()):
                if month in etm_months.get(agent, set()):
                    qlog.add("missing_acts_suspected", agent, 1,
                             f"net aktov za {month}, no ETM aktivna", dedup=True)
                else:
                    qlog.add("idle_month", agent, 1, month, dedup=True)


def check_void_sale_gaps(etm: pd.DataFrame, cfg: dict, qlog: QualityLog) -> None:
    """Voyd ran'she prodazhi: flaguem VSE sluchai s velichinoy gapa.
    gap <= edge_days — kray okna (dekabr), bolshe — alarm na revyu."""
    dcfg = cfg.get("dates", {})
    edge = pd.Timedelta(days=dcfg.get("edge_days", 45))
    sale_kinds = {k.lower() for k in cfg.get("sale_kinds", ["выкуп"])}
    void_kinds = {k.lower() for k in cfg.get("void_kinds", [])}
    earliest: dict[tuple, pd.Timestamp] = {}
    for _, row in etm.iterrows():
        if str(row.get("kind", "")).strip().lower() in sale_kinds and pd.notna(row["date_norm"]):
            for ticket in row["ticket_set"]:
                key = (row["agent_norm"], ticket)
                if key not in earliest or row["date_norm"] < earliest[key]:
                    earliest[key] = row["date_norm"]
    for _, row in etm.iterrows():
        if str(row.get("kind", "")).strip().lower() not in void_kinds:
            continue
        if pd.isna(row["date_norm"]) or not row["ticket_set"]:
            continue
        sales = [earliest[(row["agent_norm"], t)] for t in row["ticket_set"]
                 if (row["agent_norm"], t) in earliest]
        if not sales:
            qlog.add("void_without_sale", str(row["agent_norm"]), 1,
                     f"{row.get('txn_id', '?')}: {sorted(row['ticket_set'])[0]}", dedup=True)
            continue
        first_sale = min(sales)
        if row["date_norm"] < first_sale:
            gap_days = (first_sale - row["date_norm"]).days
            tag = "window_edge" if (first_sale - row["date_norm"]) <= edge else "alarm"
            worst = sorted(row["ticket_set"])[0]
            qlog.add("void_predates_sale", str(row["agent_norm"]), 1,
                     f"{row.get('txn_id', '?')}: voyd na {gap_days} dn. ran'she prodazhi "
                     f"[{tag}] ({worst})")


def check_void_full_refund(etm: pd.DataFrame, cfg: dict, qlog: QualityLog) -> None:
    """Voyd bez (polnogo) vozvrata (Shag 5): po kazhdoy voyd-stroke netto prodazh
    vnutri eyo biletov dolzhno zakryvatsya v 0 (ETM-arifmetika).
    Multi-biletnye voydy schitaem mnozhestvami (tail-privyazka kak v dengakh);
    prodazhi — vynuzhdenno shiroko (vykup+vozvrat+oplata): strogo tolko vykup
    dal by lozhnye srabotki na smeshannyh oplatnyh gruppakh. Na tekuschih
    dannyh — 0 sluchaev (606/606 zakryty); proveryaem kazhdyy mesyats zanovo."""
    tol = cfg["tolerance"]
    void_kinds = {k.lower() for k in cfg.get("void_kinds", [])}
    sale_kinds = ({k.lower() for k in cfg.get("sale_kinds", ["выкуп"])}
                  | {"возврат", "оплата"})

    def tails(ts) -> set:
        items = ts if isinstance(ts, set) else set()
        return {str(t)[-10:] for t in items if str(t).strip()}

    sales_by_agent: dict = defaultdict(list)
    for _, row in etm.iterrows():
        if str(row.get("kind", "")).strip().lower() in sale_kinds:
            sales_by_agent[row["agent_norm"]].append(
                (tails(row.get("ticket_set")), float(row.get("sum_etm", 0.0))))
    for _, row in etm.iterrows():
        if str(row.get("kind", "")).strip().lower() not in void_kinds:
            continue
        vt = tails(row.get("ticket_set"))
        if not vt:
            continue
        covered = sum(s for st, s in sales_by_agent.get(row["agent_norm"], [])
                      if st and st <= vt)
        net = round(covered + float(row.get("sum_etm", 0.0)), 2)
        if abs(net) >= tol:
            qlog.add("void_partial_refund", str(row["agent_norm"]), 1,
                     f"txn {row.get('txn_id', '?')}: voyd ne zakryl prodazhi "
                     f"(net {net})", dedup=True)


def check_balance_chain(etm: pd.DataFrame, cfg: dict, qlog: QualityLog) -> None:
    """Nepreryvnost deneg ETM: balance[i] == balance[i-1] + amount[i].
    Cherez sum_etm (rabotaet i pri konvertirovannykh valyutakh)."""
    if not cfg.get("dates", {}).get("check_balance_chain", True):
        return
    tol = cfg.get("dates", {}).get("balance_tolerance", cfg["tolerance"])
    for agent, group in etm.groupby("agent_norm"):
        ordered = group.sort_values("date_norm").reset_index()
        pred = ordered["balance_after"].shift(1) - ordered["sum_etm"]
        broken = ((ordered["balance_after"] - pred).abs() > tol) & pred.notna() \
            & ordered["balance_after"].notna() & ordered["sum_etm"].notna()
        for pos in ordered.index[broken]:
            row = ordered.loc[pos]
            qlog.add("balance_chain_break", str(agent), 1,
                     f"{row.get('txn_id', '?')}: bal {row['balance_after']:.2f} "
                     f"vs ozhid {float(pred.loc[pos]):.2f}")


def check_cross_agent_doc_refs(acts: pd.DataFrame, etm: pd.DataFrame, cfg: dict,
                               qlog: QualityLog) -> None:
    """Platezh ETM ssylaetsya na nomer, prinadlezhaschiy aktu DRUGOGO agenta.
    Obosnovanie vnutriagentnogo skopa doc-matcha: 585 nomerov kolлизируют
    mezhdu agentami (u kazhdogo svoya numeratsiya s 000001)."""
    owners: dict[str, set[str]] = defaultdict(set)
    for _, row in acts.iterrows():
        ref = doc_ref(row.get("doc", ""), cfg)
        if ref:
            owners[ref].add(str(row["agent_norm"]))
    if "comment" not in etm.columns:
        return
    for _, row in etm.iterrows():
        if pd.notna(row.get("tickets")) and str(row.get("tickets", "")).strip():
            continue
        ref = doc_ref(row["comment"], cfg)
        if not ref or ref not in owners:
            continue
        if str(row["agent_norm"]) not in owners[ref]:
            qlog.add("cross_agent_doc_ref", str(row["agent_norm"]), 1,
                     f"{row.get('txn_id', '?')} {ref} -> vladeltsy: "
                     f"{', '.join(sorted(owners[ref]))}")


def check_saldo_consistency(acts: pd.DataFrame, qlog: QualityLog) -> None:
    """Storozhevaya: gulyanie saldo vnutri papka+period posle status-filtra.
    Seychas 0 (15 fevralskikh versiy — markirovannye chernovik/pereizdan);
    srabotka = nemarkirovannoe pereizdanie (molchalivoe udvoenie)."""
    if not len(acts):
        return
    for col in ("saldo_start", "saldo_end"):
        nunique = acts.groupby(["folder", "period_start"])[col].nunique()
        for (folder, period), count in nunique[nunique > 1].items():
            spread = acts.loc[(acts["folder"] == folder)
                              & (acts["period_start"] == period), col]
            qlog.add("saldo_inconsistent", f"{folder} {period}", int(count),
                     f"{col}: {float(spread.min()):.2f}..{float(spread.max()):.2f}", dedup=True)


def check_shared_registry_tickets(reg: pd.DataFrame, qlog: QualityLog) -> None:
    """Odin bilet u 2+ storon reestra — nablyudenie dlya sito imen (vozmozhnye dubl-imen)."""
    by_ticket: dict[str, set[str]] = defaultdict(set)
    for _, row in reg.iterrows():
        for ticket in row["ticket_set"]:
            by_ticket[ticket].add(str(row["agent_norm"]))
    shared = {t: parties for t, parties in by_ticket.items() if len(parties) > 1}
    if shared:
        examples = "; ".join(f"{t}: {'|'.join(sorted(p)[:3])}"
                             for t, p in list(shared.items())[:3])
        qlog.add("shared_ticket_across_parties", "total", len(shared), examples, dedup=True)


# ------------------------------------------------------------------ main ---

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    INPUT = HERE.parent / "input"
    REPORTS = HERE.parent / "reports"
    parser.add_argument("--acts", type=Path, default=INPUT / "acts.csv")
    parser.add_argument("--etm", type=Path, default=INPUT / "etm.csv")
    parser.add_argument("--registry", type=Path, default=INPUT / "registry.csv")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=REPORTS / "discrepancies.csv")
    parser.add_argument("--summary", type=Path, default=REPORTS / "reconciliation_summary.csv")
    parser.add_argument("--quality", type=Path, default=None)
    parser.add_argument("--review-candidates", type=Path,
                        default=REPORTS / "review_candidates.csv")
    args = parser.parse_args()
    cfg = load_config(args.config)

    qlog = QualityLog()
    acts, etm, reg, ignored_drafts = load_data(args.acts, args.etm, args.registry, cfg, qlog)

    # Agenty tolko v odnom fayle — vidimost, ne padenie (summary otrabotaet vetkoy one_sided).
    only_acts = sorted(set(acts["agent_norm"]) - set(etm["agent_norm"]))
    only_etm = sorted(set(etm["agent_norm"]) - set(acts["agent_norm"]))
    for agent in only_acts:
        qlog.add("one_sided_agent", f"only_acts:{agent}", 1, "", dedup=True)
    for agent in only_etm:
        qlog.add("one_sided_agent", f"only_etm:{agent}", 1, "", dedup=True)
    orphans = reg.loc[~reg["agent_norm"].isin(set(acts["agent_norm"]) | set(etm["agent_norm"]))]
    if len(orphans):
        qlog.add("registry_orphans", "no_match_in_acts_etm", len(orphans),
                 str(orphans["party"].iloc[0]), dedup=True)

    # --- diagnostika dat i denezhnykh tsepochek (paket D1-D4) ---
    check_date_ranges(acts, etm, cfg, qlog)
    check_missing_periods(acts, etm, cfg, qlog)
    check_void_sale_gaps(etm, cfg, qlog)
    check_void_full_refund(etm, cfg, qlog)
    check_balance_chain(etm, cfg, qlog)
    check_shared_registry_tickets(reg, qlog)
    # --- paket E: chuzhie dokumenty, storozh saldo, metki agentov ---
    check_cross_agent_doc_refs(acts, etm, cfg, qlog)
    check_saldo_consistency(acts, qlog)
    check_multi_label_agents(acts, etm, qlog)

    discrepancies = pd.DataFrame(
        reconcile_tickets(acts, etm, reg, cfg, qlog) + reconcile_payments(acts, etm, cfg, qlog)
    )
    if len(discrepancies):
        discrepancies = discrepancies.sort_values(["agent", "date", "identifier"]).reset_index(drop=True)
        for column in ("sum_1c", "sum_etm", "diff"):
            discrepancies[column] = discrepancies[column].round(2)
    else:
        discrepancies = pd.DataFrame(
            columns=["agent", "date", "identifier", "sum_1c", "sum_etm", "diff", "comment",
                     "source"]
        )
    discrepancies.to_csv(args.output, index=False, encoding="utf-8-sig")

    summary = reconciliation_summary(acts, etm, discrepancies, ignored_drafts, cfg)
    numeric_columns = summary.select_dtypes(include="number").columns
    summary[numeric_columns] = summary[numeric_columns].round(2)
    summary.to_csv(args.summary, index=False, encoding="utf-8-sig")

    labels = canonical_labels(acts, etm, cfg)
    review_candidates = build_review_candidates(acts, etm, reg, cfg, labels)
    review_candidates.to_csv(args.review_candidates, index=False, encoding="utf-8-sig")

    quality_path = args.quality or (REPORTS / cfg["output"]["quality_report"])
    qlog.to_frame().to_csv(quality_path, index=False, encoding="utf-8-sig")

    print(f"Wrote {len(discrepancies)} discrepancies to {args.output}")
    print(f"Wrote balance checks for {len(summary)} agents to {args.summary}")
    print(f"Wrote {len(review_candidates)} non-authoritative review candidates to "
          f"{args.review_candidates}")
    print(f"Wrote {len(qlog.to_frame())} quality notes to {quality_path}")
    for check, total in sorted(qlog.summary().items()):
        print(f"  quality/{check}: {total}")
    if not len(summary):
        print("NOTE: pustoy summary (net dannykh)")
        return
    bad = summary.loc[~summary["operations_explain_movement"] & ~summary["one_sided"]]
    if len(bad):
        print(f"WARNING: {len(bad)} agents do not reconcile, see residual != 0")
        print(bad[["agent", "ending_gap", "explained_gap", "residual"]].to_string(index=False))
    one_sided = summary.loc[summary["one_sided"]]
    if len(one_sided):
        print(f"NOTE: {len(one_sided)} one-sided agents (net dannykh s odnoy storony)")
        print(one_sided[["agent", "alias_of"]].to_string(index=False))


if __name__ == "__main__":
    main()
