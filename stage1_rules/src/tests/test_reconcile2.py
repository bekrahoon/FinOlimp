"""Testy universalnogo reconcile2: sintetika na kazhduyu situatsiyu + regressiya.

Zapusk: pytest src/tests/ -q  (cwd = stage1_rules/)
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

SS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SS))

import reconcile2 as R  # noqa: E402

TICKET = "1234567890"


def base_cfg(**over):
    cfg = R.load_config(None)  # realnyy reconcile.yaml ryadom (defolty dvizhka)
    for dotted, value in over.items():
        node = cfg
        *path, last = dotted.split(".")
        for part in path:
            node = node[part]
        node[last] = value
    return cfg


def write_inputs(tmp: Path, acts_rows, etm_rows, reg_rows, encoding="utf-8"):
    a, e, r = tmp / "acts.csv", tmp / "etm.csv", tmp / "registry.csv"
    pd.DataFrame(acts_rows).to_csv(a, index=False, encoding=encoding)
    pd.DataFrame(etm_rows).to_csv(e, index=False, encoding=encoding)
    pd.DataFrame(reg_rows).to_csv(r, index=False, encoding=encoding)
    return a, e, r


def acts_row(folder="Test Agent", date="2026-03-05", doc="Реализация NB TEST01",
             ticket=TICKET, debet=1000.0, credit=0.0,
             saldo_start=0.0, saldo_end="auto",
             period_start="2026-01-01", period_end="2026-01-31"):
    if saldo_end == "auto":
        saldo_end = saldo_start + (debet - credit)
    return {"folder": folder, "period_start": period_start, "period_end": period_end,
            "saldo_start": saldo_start, "saldo_end": saldo_end, "date": date, "doc": doc,
            "ticket_cell": ticket, "debet": debet, "credit": credit}


def acts_pay(folder="Test Agent", date="2026-03-05", doc="Поступление на р/с, п/п ЦБ-С000001",
             credit=500.0, saldo_start=0.0, saldo_end="auto",
             period_start="2026-01-01", period_end="2026-01-31"):
    row = acts_row(folder, date, doc, None, 0.0, credit, saldo_start, saldo_end,
                   period_start, period_end)
    row["ticket_cell"] = None
    return row


def etm_row(agent="Test Agent", date="2026-03-05", kind="выкуп", tickets=TICKET,
            amount_kgs=-1000.0, balance_after=0.0, creator="etm-bot", comment="",
            txn="90000001", **extra):
    row = {"agent": agent, "txn_id": txn, "date": date, "kind": kind,
           "tickets": tickets, "amount_kgs": amount_kgs,
           "balance_after": balance_after, "creator": creator, "comment": comment}
    row.update(extra)
    return row


def etm_pay(agent="Test Agent", date="2026-03-05", amount_kgs=500.0,
            balance_after=0.0, comment="", txn="90000002", **extra):
    row = etm_row(agent, date, "оплата", None, amount_kgs, balance_after,
                  "etm-bot", comment, txn, **extra)
    row["tickets"] = None
    return row


def reg_row(party="Test Agent", tickets=TICKET, employee="a.test",
            date="05.03.2026", **extra):
    row = {"date": date, "employee": employee, "party": party, "tickets": tickets}
    row.update(extra)
    return row


def run_all(tmp, cfg, acts_rows, etm_rows, reg_rows, encoding="utf-8"):
    a, e, r = write_inputs(tmp, acts_rows, etm_rows, reg_rows, encoding)
    qlog = R.QualityLog()
    acts, etm, reg, ignored = R.load_data(a, e, r, cfg, qlog)
    # Diagnostika main(): daty, dyry, voydy, tsepochka, obschie bilety.
    R.check_date_ranges(acts, etm, cfg, qlog)
    R.check_missing_periods(acts, etm, cfg, qlog)
    R.check_void_sale_gaps(etm, cfg, qlog)
    R.check_balance_chain(etm, cfg, qlog)
    R.check_shared_registry_tickets(reg, qlog)
    # Paket E: chuzhie dokumenty, storozh saldo, metki agentov.
    R.check_cross_agent_doc_refs(acts, etm, cfg, qlog)
    R.check_saldo_consistency(acts, qlog)
    R.check_multi_label_agents(acts, etm, qlog)
    disc = pd.DataFrame(R.reconcile_tickets(acts, etm, reg, cfg, qlog)
                        + R.reconcile_payments(acts, etm, cfg, qlog))
    if len(disc) == 0:
        disc = pd.DataFrame(
            columns=["agent", "date", "identifier", "sum_1c", "sum_etm", "diff", "comment"])
    summ = R.reconciliation_summary(acts, etm, disc, ignored, cfg)
    return acts, etm, reg, disc, summ, qlog


# ------------------------------------------------------------- situatsii ---

def test_legacy_ticket_match(tmp_path):
    """Baza: sovpavshie bilety — ni stroki rashozhdeniy, residual 0."""
    _, _, _, disc, summ, _ = run_all(
        tmp_path, base_cfg(), [acts_row()], [etm_row()], [reg_row()])
    assert disc.empty
    assert float(summ.iloc[0]["residual"]) == 0.0
    assert not bool(summ.iloc[0]["one_sided"])


def test_currency_conversion_without_amount_kgs(tmp_path):
    """USD bez amount_kgs: kurs iz reestra na datu (85) -> sum_etm=-8500."""
    cfg = base_cfg()
    etm = [etm_row(amount_kgs=None, txn="90000010")]
    etm[0].pop("amount_kgs")
    etm[0]["amount"] = 100.0
    etm[0]["currency"] = "USD"
    reg = [reg_row(rate_usd=85.0, date="01.01.2026")]
    acts = [acts_row(credit=8500.0, debet=0.0)]
    _, _, _, disc, _, _ = run_all(tmp_path, cfg, acts, etm, reg)
    assert disc.empty  # soshlos => konversiya vernaya (pri rate=1 byl by diff)


def test_rate_by_date(tmp_path):
    """Kurs menyaetsya v seredine perioda: kazhdaya operatsiya po svoemu kursu."""
    reg = [reg_row(rate_usd=80.0, date="01.01.2026"),
           reg_row(rate_usd=90.0, date="01.03.2026", txn="x")]
    etm = [etm_pay(date="2026-02-01", comment="a", txn="90000011"),
           etm_pay(date="2026-04-01", comment="b", txn="90000012")]
    for row in etm:
        row.pop("amount_kgs")
        row["amount"] = 100.0
        row["currency"] = "USD"
    acts = [acts_pay(date="2026-02-01", credit=8000.0),
            acts_pay(date="2026-04-01", doc="Поступление на р/с, п/п ЦБ-С000002", credit=9000.0)]
    _, _, _, disc, _, _ = run_all(tmp_path, base_cfg(), acts, etm, reg)
    assert disc.empty


def test_agent_alias_merges_rename(tmp_path):
    """Pereimenovanie v seredine goda: alias skladyvaet v odnogo agenta."""
    cfg = base_cfg()
    cfg["agents"]["aliases"] = {"старое осоо тест": "новое осоо тест"}
    acts = [acts_row(folder="Старое ОсОО Тест")]
    etm = [etm_row(agent="Новое ОсОО Тест")]
    _, _, _, disc, summ, _ = run_all(
        tmp_path, cfg, acts, etm, [reg_row(party="Новое ОсОО Тест")])
    assert disc.empty
    assert "Старое" in str(summ.iloc[0]["alias_of"])


def test_cross_agent_fallback_flags_ticket(tmp_path):
    """Bez aliasa: tot zhe bilet u drugogo agenta -> pometka кросс-агент."""
    acts = [acts_row(folder="Agent A")]
    etm = [etm_row(agent="Agent B", amount_kgs=0.0)]
    etm[0]["amount_kgs"] = -1000.0
    _, _, _, disc, _, qlog = run_all(
        tmp_path, base_cfg(), acts, etm, [reg_row(party="Agent A")])
    assert len(disc) == 2
    assert all("кросс-агент" in str(c) for c in disc["comment"])
    assert "cross_agent_ticket" in qlog.summary()


def test_payment_window_blocks_far_match(tmp_path):
    """Odinakovye summy v 150 dnyah: default-okno 120 ne skhlepyvaet."""
    acts = [acts_pay(date="2026-01-10", credit=500.0)]
    etm = [etm_pay(date="2026-06-10", amount_kgs=500.0)]
    _, _, _, disc, _, _ = run_all(tmp_path, base_cfg(), acts, etm, [reg_row(tickets=None)])
    assert len(disc) == 2
    wide = base_cfg(**{"payments.date_window_days": 200})
    _, _, _, disc2, _, _ = run_all(tmp_path, wide, acts, etm, [reg_row(tickets=None)])
    assert disc2.empty


def test_doc_match_beats_window(tmp_path):
    """Tot zhe ЦБ-С po obe storony: tochnyy match vne zavisimosti ot okna."""
    acts = [acts_pay(date="2026-01-10", credit=500.0)]
    etm = [etm_pay(date="2026-06-10", amount_kgs=500.0, comment="п/п ЦБ-С000001")]
    _, _, _, disc, _, _ = run_all(tmp_path, base_cfg(), acts, etm, [reg_row(tickets=None)])
    assert disc.empty


def test_one_sided_agent_no_crash(tmp_path):
    """Agent tolko v ETM: summary s one_sided=True, bez IndexError."""
    acts = [acts_row(folder="Agent A")]
    etm = [etm_row(agent="Agent A"), etm_row(agent="Agent B", txn="90000020")]
    _, _, _, _, summ, _ = run_all(tmp_path, base_cfg(), acts, etm, [reg_row()])
    row_b = summ[summ["agent"] == "Agent B"].iloc[0]
    assert bool(row_b["one_sided"])
    assert pd.isna(row_b["residual"])


def test_void_remainder_optin(tmp_path):
    """Voyd s ostatkom: defolt — obschiy diagnoz; flag — otdelnyy."""
    acts = [acts_row(debet=1000.0)]
    etm = [etm_row(amount_kgs=-1000.0, txn="90000030"),
           etm_row(kind="войд", amount_kgs=200.0, txn="90000031")]
    _, _, _, disc, _, _ = run_all(tmp_path, base_cfg(), acts, etm, [reg_row()])
    assert len(disc) == 1 and "Суммы операции отличаются" in str(disc.iloc[0]["comment"])
    cfg = base_cfg(**{"comments.void_remainder_note": True})
    _, _, _, disc2, _, _ = run_all(tmp_path, cfg, acts, etm, [reg_row()])
    assert "Войд" in str(disc2.iloc[0]["comment"])


def test_cp1251_files(tmp_path):
    """Fayly v cp1251 s kirillitsey chitayutsya i veryatsya."""
    acts = [acts_row(folder="Тестовый Агент")]
    etm = [etm_row(agent="Тестовый Агент")]
    _, _, _, disc, summ, _ = run_all(
        tmp_path, base_cfg(), acts, etm, [reg_row(party="Тестовый Агент")],
        encoding="cp1251")
    assert disc.empty
    assert float(summ.iloc[0]["residual"]) == 0.0


def test_summary_balances_with_payment(tmp_path):
    """Polnyy tsikl: bilet + platezh s dochnyim matchem -> residual 0."""
    acts = [acts_row(saldo_start=100.0, saldo_end=600.0),
            acts_pay(doc="Поступление на р/с, п/п ЦБ-С000007", credit=500.0)]
    etm = [etm_row(balance_after=-600.0, txn="90000040"),
           etm_pay(comment="п/п ЦБ-С000007", amount_kgs=500.0,
                   balance_after=-100.0, txn="90000041")]
    _, _, _, disc, summ, _ = run_all(tmp_path, base_cfg(), acts, etm, [reg_row()])
    assert disc.empty
    assert float(summ.iloc[0]["residual"]) == 0.0
    assert bool(summ.iloc[0]["operations_explain_movement"])


# ------------------------------------------------------------------- daty ---

def test_tz_mixed_no_crash(tmp_path):
    """Smes naive + +06:00: bez TypeError, flag tz_normalized, match rabotaet."""
    acts = [acts_row()]
    etm = [etm_row(date="2026-03-05 10:00:00+06:00")]
    _, _, _, disc, _, qlog = run_all(tmp_path, base_cfg(), acts, etm, [reg_row()])
    assert disc.empty
    assert "tz_normalized" in qlog.summary()


def test_date_out_of_range(tmp_path):
    """Opechatka goda (2036): stroka odnostoronnyaya + flag date_out_of_range."""
    acts = [acts_pay(date="2036-01-05")]
    etm = [etm_pay(date="2026-01-05")]
    _, _, _, disc, _, qlog = run_all(tmp_path, base_cfg(), acts, etm, [reg_row(tickets=None)])
    assert len(disc) == 2
    assert "date_out_of_range" in qlog.summary()


def test_missing_month_suspected_and_idle(tmp_path):
    """Dyra s ETM-aktivnostyu — krasnyy flag; bez nee — tikhiy prostoy."""
    acts = [acts_pay(folder="Agent A", date="2026-01-10"),
            acts_pay(folder="Agent A", date="2026-03-10",
                     doc="Поступление на р/с, п/п ЦБ-С000002",
                     period_start="2026-03-01", period_end="2026-03-31"),
            acts_pay(folder="Agent B", date="2026-01-10"),
            acts_pay(folder="Agent B", date="2026-03-10",
                     doc="Поступление на р/с, п/п ЦБ-С000003",
                     period_start="2026-03-01", period_end="2026-03-31")]
    etm = [etm_pay(agent="Agent A", date="2026-01-10"),
           etm_pay(agent="Agent A", date="2026-02-10", txn="90000201"),
           etm_pay(agent="Agent A", date="2026-03-10", comment="п/п ЦБ-С000002", txn="90000202"),
           etm_pay(agent="Agent B", date="2026-01-10", txn="90000203"),
           etm_pay(agent="Agent B", date="2026-03-10",
                   comment="п/п ЦБ-С000003", txn="90000204")]
    _, _, _, _, _, qlog = run_all(tmp_path, base_cfg(), acts, etm, [reg_row(tickets=None)])
    summary = qlog.summary()
    assert "missing_acts_suspected" in summary
    assert "idle_month" in summary
    notes = qlog.to_frame()
    assert "agent a" in str(notes.loc[notes["check"] == "missing_acts_suspected", "key"].tolist())
    assert "agent b" in str(notes.loc[notes["check"] == "idle_month", "key"].tolist())


def test_void_predates_sale_edge_and_alarm(tmp_path):
    """Voyd za 10 dney do prodazhi — window_edge; za 100 — alarm."""
    t1, t2 = "7777777777", "8888888888"
    acts = [acts_row(ticket=t1, debet=1000.0), acts_row(ticket=t2, debet=2000.0)]
    etm = [etm_row(tickets=t1, date="2026-03-20", amount_kgs=-1000.0, txn="90000301"),
           etm_row(tickets=t1, date="2026-03-10", kind="войд", amount_kgs=100.0, txn="90000302"),
           etm_row(tickets=t2, date="2026-05-20", amount_kgs=-2000.0, txn="90000303"),
           etm_row(tickets=t2, date="2026-02-09", kind="войд", amount_kgs=200.0, txn="90000304")]
    _, _, _, _, _, qlog = run_all(tmp_path, base_cfg(), acts, etm, [reg_row()])
    notes = qlog.to_frame()
    void_notes = notes.loc[notes["check"] == "void_predates_sale", "sample"].tolist()
    assert len(void_notes) == 2
    assert any("[window_edge]" in s for s in void_notes)
    assert any("[alarm]" in s for s in void_notes)


def test_void_without_sale(tmp_path):
    """Voyd po biletu bez vykupa — otdelnyy flag."""
    etm = [etm_row(tickets="9999999999", kind="войд", amount_kgs=100.0, txn="90000305")]
    _, _, _, _, _, qlog = run_all(
        tmp_path, base_cfg(), [acts_row()], etm, [reg_row()])
    assert "void_without_sale" in qlog.summary()


def test_balance_chain_break(tmp_path):
    """Razryv tsepochki balansa: flag s faktom i ozhidaniem, bez padeniya."""
    acts = [acts_pay(credit=500.0)]
    etm = [etm_pay(amount_kgs=500.0, balance_after=500.0, txn="90000401"),
           etm_pay(amount_kgs=200.0, balance_after=700.0, txn="90000402"),
           etm_pay(amount_kgs=50.0, balance_after=1000.0, txn="90000403")]
    _, _, _, _, summ, qlog = run_all(tmp_path, base_cfg(), acts, etm, [reg_row(tickets=None)])
    assert "balance_chain_break" in qlog.summary()
    assert len(summ) == 1  # summary ne padaet


def test_payment_tiebreak_logged(tmp_path):
    """Dva kandidata na ravnykh distantsiyakh: vybran pervyy + sled v quality."""
    acts = [acts_pay(date="2026-03-10", credit=500.0)]
    etm = [etm_pay(date="2026-03-08", amount_kgs=500.0, txn="90000501"),
           etm_pay(date="2026-03-12", amount_kgs=500.0, txn="90000502")]
    _, _, _, disc, _, qlog = run_all(tmp_path, base_cfg(), acts, etm, [reg_row(tickets=None)])
    assert len(disc) == 1  # odin sparilsya, vtoroy ostalsya — i eto zafiksirovano
    assert "payment_tiebreak" in qlog.summary()
    sample = qlog.to_frame().loc[qlog.to_frame()["check"] == "payment_tiebreak", "sample"].iloc[0]
    assert "90000502" in sample  # otbroshennyy kandidat zafiksirovan


def test_shared_ticket_and_odd_money(tmp_path):
    """Obschiy bilet u 2 storon + dvustoronnyaya stroka deneg."""
    acts = [acts_row(debet=100.0, credit=50.0)]
    etm = [etm_row()]
    reg = [reg_row(party="Test Agent"), reg_row(party="Test Agent-2", employee="b.test")]
    _, _, _, _, _, qlog = run_all(tmp_path, base_cfg(), acts, etm, reg)
    assert "shared_ticket_across_parties" in qlog.summary()
    assert "odd_money_sign" in qlog.summary()

# --------------------------------------------------------------------- E ---

def empty_inputs(tmp):
    acts_cols = ["folder", "period_start", "period_end", "saldo_start", "saldo_end",
                 "date", "doc", "ticket_cell", "debet", "credit"]
    etm_cols = ["agent", "txn_id", "date", "kind", "tickets", "amount_kgs",
                "balance_after", "creator", "comment"]
    reg_cols = ["date", "employee", "party", "tickets"]
    paths = [tmp / "acts.csv", tmp / "etm.csv", tmp / "registry.csv"]
    for path, cols in zip(paths, (acts_cols, etm_cols, reg_cols)):
        pd.DataFrame(columns=cols).to_csv(path, index=False)
    return paths[0], paths[1], paths[2]


def test_empty_inputs_no_crash(tmp_path):
    """Pustye fayly (tolko shapka): bez isklyucheniy, pustye vykhody."""
    a, e, r = empty_inputs(tmp_path)
    qlog = R.QualityLog()
    cfg = base_cfg()
    acts, etm, reg, ignored = R.load_data(a, e, r, cfg, qlog)
    R.check_date_ranges(acts, etm, cfg, qlog)
    R.check_missing_periods(acts, etm, cfg, qlog)
    R.check_saldo_consistency(acts, qlog)
    R.check_multi_label_agents(acts, etm, qlog)
    disc = pd.DataFrame(columns=["agent", "date", "identifier", "sum_1c",
                                 "sum_etm", "diff", "comment"])
    summ = R.reconciliation_summary(acts, etm, disc, ignored, cfg)
    assert disc.empty and summ.empty


def test_cross_agent_doc_ref(tmp_path):
    """Ssylka na nomer chuzhogo agenta: flag s ukazaniem vladeltsa."""
    acts = [acts_pay(folder="Agent A", doc="Поступление на р/с, п/п ЦБ-С000001", credit=500.0)]
    etm = [etm_pay(agent="Agent B", comment="п/п ЦБ-С000001", amount_kgs=500.0)]
    _, _, _, _, _, qlog = run_all(tmp_path, base_cfg(), acts, etm, [reg_row(tickets=None)])
    notes = qlog.to_frame()
    hits = notes.loc[notes["check"] == "cross_agent_doc_ref"]
    assert len(hits) == 1
    assert "agent a" in str(hits.iloc[0]["sample"])  # vladelets ukazan


def test_saldo_inconsistent_tripwire(tmp_path):
    """Gulyanie saldo vnutri papka+period: flag (buduschie nemarkirovannye versii)."""
    rows = [acts_row(saldo_start=0.0, saldo_end=1000.0),
            acts_row(date="2026-01-02", doc="Реализация NB TEST02",
                     ticket="2222222222", debet=500.0,
                     saldo_start=0.0, saldo_end=900.0)]
    _, _, _, _, _, qlog = run_all(tmp_path, base_cfg(), rows, [etm_row()], [reg_row()])
    assert "saldo_inconsistent" in qlog.summary()


def test_reissued_versions_single_saldo(tmp_path):
    """Fe validskie versii: chernovik vybroshen, storozh molchit."""
    v1 = acts_row(ticket="3333333333", debet=100.0, saldo_start=0.0, saldo_end=100.0)
    v2 = dict(v1, saldo_end=150.0)
    acts = [v1, v2]
    a, e, r = write_inputs(tmp_path, acts, [etm_row()], [reg_row()])
    # Ruchnaya razmetka statusov kak v realnyh fevralskikh versiyakh:
    import csv as _csv
    with open(a, encoding="utf-8") as fh:
        reader = list(_csv.DictReader(fh))
    reader[0]["act_status"] = "черновик"
    reader[1]["act_status"] = "переиздан"
    with open(a, "w", encoding="utf-8", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=list(reader[0].keys()))
        writer.writeheader()
        writer.writerows(reader)
    cfg = base_cfg()
    qlog = R.QualityLog()
    acts_ld, _, _, ignored = R.load_data(a, e, r, cfg, qlog)
    assert len(acts_ld) == 1  # chernovik vybroshen
    R.check_saldo_consistency(acts_ld, qlog)
    assert "saldo_inconsistent" not in qlog.summary()
    assert sum(ignored.values()) == 1


def test_multi_label_flag(tmp_path):
    """Dve syrye podpisi vnutri storony: flag, schet ne padaet."""
    acts = [acts_row(folder="Test Agent"),
            acts_row(folder="Test Agent LLC", date="2026-03-06",
                     doc="Реализация NB TEST02", ticket="4444444444", debet=200.0)]
    etm = [etm_row(), etm_row(tickets="4444444444", amount_kgs=-200.0, txn="90000602")]
    _, _, _, disc, _, qlog = run_all(tmp_path, base_cfg(), acts, etm, [reg_row()])
    assert disc.empty
    assert "multi_label_agent" in qlog.summary()


def test_config_validation_warning(tmp_path, capsys):
    """Bit yy yaml: WARNING, a ne padenie; rabotayut defolty."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("tolerance: mnogo\npayments: {date_window_days: 120}\nneizvestnyy_razdel: {a: 1}\n",
                   encoding="utf-8")
    cfg = R.load_config(bad)
    out = capsys.readouterr().out
    assert "WARNING (config)" in out and "unknown config key" in out
    assert cfg["tolerance"] == 0.01  # nevernyy tip -> default
    assert cfg["payments"]["date_window_days"] == 120


def test_canonical_labels_switch(tmp_path):
    """most_frequent vybiraet chastuyu podpis; default acts_first ne trogut."""
    acts = [acts_row(folder="Test Agent")]
    etm = [etm_row(agent="TEST AGENT"), etm_row(agent="TEST AGENT", tickets="5555555555",
                                                amount_kgs=-50.0, txn="90000702")]
    _, _, _, _, summ, _ = run_all(tmp_path, base_cfg(), acts, etm, [reg_row()])
    assert summ.iloc[0]["agent"] == "Test Agent"
    freq = base_cfg(**{"labels.source": "most_frequent"})
    _, _, _, _, summ2, _ = run_all(tmp_path, freq, acts, etm, [reg_row()])
    assert summ2.iloc[0]["agent"] == "TEST AGENT"

def test_dirty_money_strings_no_crash(tmp_path):
    """Syrye stroki v debet/credit/saldo: chistyatsya tsentralno, bez TypeError."""
    acts = [acts_row(debet=1000.0)]
    acts[0]["debet"] = "1 000,00"
    acts[0]["saldo_start"] = "(500,00)"
    etm = [etm_row()]
    _, _, _, disc, _, qlog = run_all(tmp_path, base_cfg(), acts, etm, [reg_row()])
    assert disc.empty  # 1000 == 1000 posle chistki
    assert "unparseable_money" not in qlog.summary()


def test_all_garbage_dates_no_crash(tmp_path):
    """Kolonka dat iz odnogo musora: NaT + flag, bez IndexError (bug strip_timezone)."""
    acts = [acts_pay(date="32.13.2026")]
    etm = [etm_pay(date="tuda-ne-znayu-kuda")]
    _, _, _, _, _, qlog = run_all(tmp_path, base_cfg(), acts, etm, [reg_row(tickets=None)])
    assert "unparseable_date" in qlog.summary()


# -------------------------------------------------------------- regressiya ---

GOLDEN_DISC_MD5 = "b2bdf894cc78d1a8d38d4ae64683d81c"  # v3 RA: agent-klass (reestr~=ETM -> agent)
GOLDEN_SUMMARY_MD5 = "9af63dbc4e5e3f93f959199046a9c51d"


def test_regression_real_data(tmp_path):
    """Realnye dannye: 2165 strok, residual 0 vezde, md5 discrepancies fiksirovan."""
    output = tmp_path / "discrepancies.csv"
    summary = tmp_path / "reconciliation_summary.csv"
    quality = tmp_path / "quality_report.csv"
    review = tmp_path / "review_candidates.csv"
    r = subprocess.run(
        [sys.executable, str(SS / "reconcile2.py"),
         "--output", str(output), "--summary", str(summary),
         "--quality", str(quality), "--review-candidates", str(review)],
        capture_output=True, text=True, cwd=str(SS),
    )
    assert r.returncode == 0, r.stderr[-1000:]
    disc = pd.read_csv(output)
    summ = pd.read_csv(summary)
    qual = pd.read_csv(quality)
    assert len(disc) == 2165
    assert (summ["residual"].abs() < 0.01).all()
    assert (summ["one_sided"] == False).all()  # noqa: E712
    digest = hashlib.md5(output.read_bytes()).hexdigest()
    assert digest == GOLDEN_DISC_MD5
    sdigest = hashlib.md5(summary.read_bytes()).hexdigest()
    assert sdigest == GOLDEN_SUMMARY_MD5  # dengi sloy imen ne tronul
    assert not any(str(c).startswith("unknown_currency") for c in qual["check"])
    assert not any(str(c).startswith("unknown_status") for c in qual["check"])
    orphans = qual.loc[qual["check"] == "registry_orphans", "count"]
    assert int(orphans.sum()) == 481  # bylo 1062, 581 spaseno subset-privyazkoy (v2)
    nospace = qual.loc[qual["check"] == "registry_nospace_hit", "count"]
    assert int(nospace.sum()) == 959
    # Paket D (daty): diagnostika — tolko nablyudenie, chisla te zhe (md5 vyshe).
    diag = qual.loc[qual["check"] == "void_predates_sale", "count"]
    assert int(diag.sum()) == 186
    holes = qual.loc[qual["check"] == "missing_acts_suspected", "count"]
    assert int(holes.sum()) == 18
    assert int(qual.loc[qual["check"] == "balance_chain_break", "count"].sum()) == 3
    assert int(qual.loc[qual["check"] == "payment_tiebreak", "count"].sum()) == 16
    shared = qual.loc[qual["check"] == "shared_ticket_across_parties", "count"]
    assert int(shared.sum()) == 15  # bylo 126: chast byla artefaktom orphan-partiy (v2)
    # v2 subset-privyazka: 581 stroka reestra prishli k agentam, 58 — dvoystvennye.
    subset = qual.loc[qual["check"] == "registry_subset_hit", "count"]
    assert int(subset.sum()) == 581
    amb = qual.loc[qual["check"] == "ambiguous_subset", "count"]
    assert int(amb.sum()) == 58
    # Paket E: chuzhie ssylki est, storozha molchat (0 sluchaev), md5 te zhe.
    cross = qual.loc[qual["check"] == "cross_agent_doc_ref", "count"]
    assert int(cross.sum()) == 29
    assert "saldo_inconsistent" not in set(qual["check"])
    assert "multi_label_agent" not in set(qual["check"])
    # Shag 5: voyd bez vozvrata — strozh, sluchaev 0 (606/606 zakryty).
    assert "void_partial_refund" not in set(qual["check"])
    # Shag 4: source kolonka — strogo iz {bot, agent, 1C, vyiyasnit}.
    assert set(disc["source"]) <= {"бот", "агент", "1С", "выяснить"}
    assert int(disc["source"].isna().sum()) == 0
    src = disc["source"].value_counts().to_dict()
    assert src.get("выяснить", 0) == 820  # -56 RA (v3 agent-klass)
    assert src.get("1С", 0) == 1022
    assert src.get("бот", 0) == 267
    assert src.get("агент", 0) == 56  # v3 RA: reestr~=ETM (S110/S117/S119/S122-klass)
    # v3 mikroruly: suffiksy v kommentariyakh (auditability; prefix ne tronut).
    assert int(disc["comment"].str.contains("v3 R1").sum()) == 107
    assert int(disc["comment"].str.contains("v3 R3a").sum()) == 77
    assert int(disc["comment"].str.contains("v3 R3b").sum()) == 24
    assert int(disc["comment"].str.contains("v3 RA").sum()) == 56


# ------------------------------------------------------------------ imena ---

def test_source_mapping_prefixes():
    """Kazhdyy prefix kommentariya -> rovno odin source; neizvestnyy -> vyiyasnit."""
    cases = {
        "Есть в 1С и реестре, нет в ETM; вероятно, бот пропустил строку": "бот",
        "Есть в ETM (бот), но нет в 1С и в реестре; проверить запись агента": "бот",
        "Суммы отличаются; в ETM вероятен дубль операции": "бот",
        "Есть только в ETM (бот); вероятно, операция не отражена в 1С": "1С",
        "Есть только в ETM (самовыписка субагента); вероятно, не отражено в 1С": "1С",
        "Операция аннулирована в ETM, но осталась в 1С; проверить 1С": "1С",
        "Платеж есть только в 1С; вероятно, не зачислен в ETM": "выяснить",
        "Платеж есть только в ETM; вероятно, не отражен в 1С или задублирован": "выяснить",
        "Суммы операции отличаются; требуется проверка первичных документов": "выяснить",
        "Есть только в 1С; вероятно, операция не проведена в ETM": "выяснить",
        "Войд в ETM закрыт частично, остаток 1.00; проверить комиссию/пересчет": "выяснить",
    }
    for comment, expected in cases.items():
        assert R.source_for_comment(comment + " (реестр: a.toktosunova)") == expected
        assert R.source_for_comment(comment + "; в реестре нет") == expected
    assert R.source_for_comment("Sovershenno novyy shablon iz buduschego") == "выяснить"

def test_registry_nospace_rescue(tmp_path):
    """Reestr 'TestAgent' -> yadro 'Test Agent': flag 'v reestre est'."""
    acts = [acts_row(debet=1000.0)]
    etm = [etm_row(amount_kgs=-999.0)]  # diff 1.0 — stroka nablyudaema
    reg = [reg_row(party="TestAgent")]
    _, _, _, disc, _, qlog = run_all(tmp_path, base_cfg(), acts, etm, reg)
    assert len(disc) == 1
    assert "в реестре есть" in str(disc.iloc[0]["comment"])
    assert "registry_nospace_hit" in qlog.summary()
    off = base_cfg(**{"names.enabled": False})
    _, _, _, disc_off, _, _ = run_all(tmp_path, off, acts, etm, reg)
    assert "в реестре нет" in str(disc_off.iloc[0]["comment"])


def test_registry_homoglyph_rescue(tmp_path):
    """Latinskaya A v kirillicheskom imeni: fold vozvraschaet match."""
    acts = [acts_row(folder="Тест Агент", debet=1000.0)]
    etm = [etm_row(agent="Тест Агент", amount_kgs=-999.0)]
    reg = [reg_row(party="Тест \u0041гент")]  # latin A
    _, _, _, disc, _, qlog = run_all(
        tmp_path, base_cfg(), acts, etm, reg)
    assert len(disc) == 1
    assert "в реестре есть" in str(disc.iloc[0]["comment"])
    assert "registry_homoglyph_hit" in qlog.summary()


def test_fuzzy_suggest_no_merge(tmp_path):
    """Opechatka 0.900 + obschie bilety: strogogo pravila net — tolko podskazka."""
    CFG_TICKET = "3333333333"
    acts = [acts_row(ticket=CFG_TICKET, debet=1000.0)]
    etm = [etm_row(tickets=CFG_TICKET, amount_kgs=-999.0)]
    reg = [reg_row(party="Test Agnet", tickets=CFG_TICKET)]  # ratio 0.900
    _, _, _, disc, _, qlog = run_all(tmp_path, base_cfg(), acts, etm, reg)
    assert "в реестре нет" in str(disc.iloc[0]["comment"])  # ne smёrdzhilos
    assert "possible_duplicate_agents" in qlog.summary()


def test_fuzzy_auto_merge_strict(tmp_path):
    """Opechatka 0.952 + obschie bilety: auto-merge; pri auto_merge=false — orphan."""
    acts = [acts_row(ticket="4444444444", debet=1000.0)]
    etm = [etm_row(tickets="4444444444", amount_kgs=-999.0)]
    reg = [reg_row(party="Test Agents", tickets="4444444444")]  # ratio 0.952
    _, _, _, disc, _, qlog = run_all(tmp_path, base_cfg(), acts, etm, reg)
    assert "в реестре есть" in str(disc.iloc[0]["comment"])
    assert "registry_fuzzy_hit" in qlog.summary()
    off = base_cfg(**{"names.auto_merge": False})
    _, _, _, disc_off, _, qlog_off = run_all(tmp_path, off, acts, etm, reg)
    assert "в реестре нет" in str(disc_off.iloc[0]["comment"])
    assert "possible_duplicate_agents" in qlog_off.summary()


def test_nospace_collision_ambiguity(tmp_path):
    """Dva agenta na odin nospace-klyuch: reestr ne mёrdzhim, flag nem."""
    t1, t2 = "5555555555", "6666666666"
    acts = [acts_row(folder="AB CD River", ticket=t1, debet=100.0),
            acts_row(folder="ABCD River", ticket=t2, debet=200.0)]
    etm = [etm_row(agent="AB CD River", tickets=t1, amount_kgs=-100.0, txn="90000101"),
           etm_row(agent="ABCD River", tickets=t2, amount_kgs=-200.0, txn="90000102")]
    reg = [reg_row(party="ABC D River", tickets=t2)]  # nospace-kolliziya
    _, _, _, disc, _, qlog = run_all(tmp_path, base_cfg(), acts, etm, reg)
    assert disc.empty  # vse soshlos po strogim klyucham
    assert "ambiguous_nospace" in qlog.summary()


def test_money_strict_despite_nospace(tmp_path):
    """ETM 'TestAgent' vs akty 'Test Agent': dengi NE skлеиvayutsya, tolko pometka."""
    acts = [acts_row(folder="Test Agent")]
    etm = [etm_row(agent="TestAgent")]
    _, _, _, disc, summ, _ = run_all(
        tmp_path, base_cfg(), acts, etm, [reg_row(party="Test Agent")])
    assert len(disc) == 2  # dve odnostoronnie stroki, a ne 0
    assert all("кросс-агент" in str(c) for c in disc["comment"])
    assert len(summ) == 2 and bool(summ["one_sided"].all())


# ------------------------------------------------------------------ Q2/Q1 ---

def test_parse_pay_kgs_total():
    """Konversiya pay_cell (Q2): yakorya S110/S112/S119 + minus + NaN-sluchai."""
    import pandas as pd

    def rates(usd=88.36, eur=95.16, rub=1.01, kzt=0.1675):
        return pd.Series({"rate_usd": usd, "rate_eur": eur,
                          "rate_rub": rub, "rate_kzt": kzt})

    assert R.parse_pay_kgs_total("303.63 usd", rates()) == 26828.75  # S110
    r112 = rates(usd=89.39, kzt=0.1808)
    assert R.parse_pay_kgs_total("KZT 135 511.00 + 5 usd sf", r112) == 24947.34  # S112
    r119 = rates(usd=87.15)
    assert R.parse_pay_kgs_total("205,98 usd + 300 kgs sf", r119) == 18251.16  # S119
    assert R.parse_pay_kgs_total("-74 088.00 rub", rates()) == -74828.88  # vozvrat
    assert R.parse_pay_kgs_total("43030 KGS + 800kgs sf", rates()) == 43830.00
    assert pd.isna(R.parse_pay_kgs_total("3762459527703", rates()))  # ssylka na bilet
    assert pd.isna(R.parse_pay_kgs_total("", rates()))
    assert pd.isna(R.parse_pay_kgs_total("100 tufrikov", rates()))  # nevalyuta
    assert pd.isna(R.parse_pay_kgs_total("100 usd", pd.Series({})))  # net kursa


def test_resolve_data_end_auto_and_explicit():
    """data_end: auto = max dat acts+etm; yavnaya data respected; pustota -> None."""
    import pandas as pd

    cfg_auto = R.load_config(None)
    assert str(cfg_auto["v3"]["data_end"]).lower() == "auto"
    acts = pd.DataFrame({"date_norm": pd.to_datetime(["2026-08-15", "2026-08-31"])})
    etm = pd.DataFrame({"date_norm": pd.to_datetime(["2026-08-20"])})
    assert R.resolve_data_end(cfg_auto, acts, etm) == pd.Timestamp("2026-08-31")
    cfg_fix = R.load_config(None)
    cfg_fix["v3"]["data_end"] = "2026-07-31"
    assert R.resolve_data_end(cfg_fix, acts, etm) == pd.Timestamp("2026-07-31")
    empty = pd.DataFrame({"date_norm": pd.to_datetime([])})
    assert R.resolve_data_end(cfg_auto, empty, empty) is None
