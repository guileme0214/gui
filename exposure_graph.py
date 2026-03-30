"""
Aggregate Exposure & Returns graph for all sweet-spot bots.

For each of the 46 bots with valid sweet spots:
  1. Re-detect the best sweet-spot filter
  2. Collect only the ops that pass that filter
  3. Per op compute:
       BACK  → exposure = matchedSize
       LAY   → exposure = matchedSize × (matchedOdd − 1)   (the liability)
       return = exposure + pl   if win (pl > 0)
              = 0               if loss

Then plot over time:
  • Cumulative invested (blue area) — total capital put at risk
  • Cumulative returned (green area) — total capital recovered
  • Cumulative net P&L line (white)
  • Bottom panel: weekly net cash flow bars
"""

import requests, time, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from datetime import datetime, timezone, timedelta
from collections import defaultdict

# ── config ───────────────────────────────────────────────────────────────────
BOTS_URL   = "https://bot.bolsadeaposta.bet.br/api/project/bots"
OPS_URL    = "https://bot.bolsadeaposta.bet.br/api/project/bots/{bot_id}/operations"
COMMISSION = 4.5
PAGE_SIZE  = 50
MIN_OPS    = 8
MIN_R2     = 0.45
MAX_DD_PCT = 0.55

SWEET_SPOT_BOT_IDS = [
    144788, 144783, 101219, 101202,
    35714, 35709, 35706, 35705, 35703, 35696, 35693, 35690, 35689,
    35677, 35675, 35310, 35306, 35302, 35298, 35296, 35293, 35288,
    21822, 21820, 21811, 21806, 21805, 21793, 21765, 21763, 21762, 21759,
    17900, 17899, 17898,
    9328, 9325, 9173,
    2748, 1968, 1868, 1438, 1334, 1333, 1331, 395,
]

SESSION = requests.Session()
SESSION.headers["Accept"] = "application/json"


# ── bucket helpers (same as sweetspot_analysis.py) ───────────────────────────
def bucket_odd(v):
    if v is None: return None
    if v < 2:  return "odd<2"
    if v < 3:  return "2≤odd<3"
    if v < 5:  return "3≤odd<5"
    if v < 8:  return "5≤odd<8"
    if v < 12: return "8≤odd<12"
    if v < 18: return "12≤odd<18"
    if v < 25: return "18≤odd<25"
    if v < 40: return "25≤odd<40"
    return "odd≥40"

def bucket_minute(v):
    if v is None: return None
    if v < 0:  return "pre-match"
    if v < 15: return "0-14'"
    if v < 30: return "15-29'"
    if v < 45: return "30-44'"
    if v < 60: return "45-59'"
    if v < 75: return "60-74'"
    if v < 90: return "75-89'"
    return "90+'"

def bucket_score(h, a):
    if h is None or a is None: return None
    if h == 0 and a == 0: return "0-0"
    if h > a:  return "home_winning"
    if a > h:  return "away_winning"
    return "level>0"

def bucket_liquidity(v):
    if v is None: return None
    if v < 10_000:    return "liq<10k"
    if v < 50_000:    return "10k≤liq<50k"
    if v < 200_000:   return "50k≤liq<200k"
    if v < 1_000_000: return "200k≤liq<1M"
    return "liq≥1M"

def bucket_period(v):
    if v is None: return None
    return f"period={v}"

BUCKET_FUNCS = {
    "odd":       lambda f: bucket_odd(f["matchedOdd"]),
    "minute":    lambda f: bucket_minute(f["minute"]),
    "score":     lambda f: bucket_score(f["homeScore"], f["awayScore"]),
    "liquidity": lambda f: bucket_liquidity(f["marketTotalMatched"]),
    "period":    lambda f: bucket_period(f["period"]),
    "selection": lambda f: f["selectionName"] or None,
}


# ── fetch ─────────────────────────────────────────────────────────────────────
def fetch_all_ops(bot_id):
    ops, page = [], 1
    while True:
        try:
            r = SESSION.get(
                OPS_URL.format(bot_id=bot_id),
                params={"page": page, "pageSize": PAGE_SIZE,
                        "sortBy": "placedAt", "sortOrder": "asc",
                        "commission": COMMISSION},
                timeout=20,
            )
            r.raise_for_status()
        except Exception as e:
            print(f"  ⚠ bot {bot_id} page {page}: {e}")
            break
        body = r.json()
        data = body.get("data", {})
        items = data.get("operations", [])
        if not items:
            break
        ops.extend(items)
        if page >= data.get("totalPages", 1):
            break
        page += 1
        time.sleep(0.1)
    return ops


def extract_features(ops):
    """Return list of rich feature dicts (one per op, sorted by date)."""
    rows = []
    for op in ops:
        bets = op.get("bets", [])
        if not bets:
            continue
        b = bets[0]
        placed = op.get("placedAt", "")
        if not placed:
            continue
        side         = b.get("side", "BACK")
        matched_size = b.get("matchedSize") or 0
        matched_odd  = b.get("matchedOdd") or 1
        pl           = op.get("pl", 0) or 0

        # exposure = capital at risk
        if side == "LAY":
            exposure = matched_size * max(matched_odd - 1, 0)
        else:
            exposure = matched_size

        # money that comes back
        returned = (exposure + pl) if pl > 0 else 0.0

        rows.append({
            "placedAt":           placed,
            "pl":                 pl,
            "exposure":           exposure,
            "returned":           returned,
            "matchedOdd":         matched_odd,
            "minute":             b.get("minute"),
            "homeScore":          b.get("homeScore"),
            "awayScore":          b.get("awayScore"),
            "period":             b.get("period"),
            "marketTotalMatched": b.get("marketTotalMatched"),
            "selectionName":      b.get("selectionName", ""),
        })
    rows.sort(key=lambda r: r["placedAt"])
    return rows


def score_slice(rows):
    n = len(rows)
    if n < MIN_OPS:
        return {"valid": False}
    pl_vals = [r["pl"] for r in rows]
    cum = np.cumsum(pl_vals)
    if cum[-1] <= 0:
        return {"valid": False}
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, cum, 1)
    if slope <= 0:
        return {"valid": False}
    ss_res = np.sum((cum - (slope * x + intercept)) ** 2)
    ss_tot = np.sum((cum - cum.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
    if r2 < MIN_R2:
        return {"valid": False}
    peak = np.maximum.accumulate(cum)
    dd_pct = np.max(peak - cum) / peak[-1] if peak[-1] > 0 else 1.0
    if dd_pct > MAX_DD_PCT:
        return {"valid": False}
    wins = sum(1 for p in pl_vals if p > 0)
    return {
        "valid": True, "n": n, "total_pl": float(cum[-1]),
        "r2": r2, "dd_pct": dd_pct, "win_rate": wins / n,
    }


def best_sweet_spot(rows):
    """Return (label, matching_rows) for the best sweet spot, or None."""
    results = []

    for dim, fn in BUCKET_FUNCS.items():
        groups = defaultdict(list)
        for r in rows:
            v = fn(r)
            if v is not None:
                groups[v].append(r)
        for v, grp in groups.items():
            sc = score_slice(grp)
            if sc["valid"]:
                results.append((f"{dim}={v}", grp, sc))

    combos = [
        ("odd","minute"), ("odd","score"), ("odd","liquidity"),
        ("minute","score"), ("odd","period"),
    ]
    for d1, d2 in combos:
        fn1, fn2 = BUCKET_FUNCS[d1], BUCKET_FUNCS[d2]
        groups = defaultdict(list)
        for r in rows:
            v1, v2 = fn1(r), fn2(r)
            if v1 is not None and v2 is not None:
                groups[(v1, v2)].append(r)
        for (v1, v2), grp in groups.items():
            sc = score_slice(grp)
            if sc["valid"]:
                results.append((f"{d1}={v1} & {d2}={v2}", grp, sc))

    if not results:
        return None, []

    def rank(item):
        _, _, sc = item
        return sc["total_pl"] * sc["r2"] * sc["win_rate"] / (sc["dd_pct"] + 0.01)

    results.sort(key=rank, reverse=True)
    label, grp, sc = results[0]
    return label, grp


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print("Fetching bot names …")
    r = SESSION.get(BOTS_URL, timeout=20)
    body = r.json()
    raw = body.get("data", body) if isinstance(body, dict) else body
    if isinstance(raw, list):
        bot_names = {b["id"]: b.get("name", str(b["id"])) for b in raw}
    else:
        bots = raw.get("bots", raw.get("items", []))
        bot_names = {b["id"]: b.get("name", str(b["id"])) for b in bots}

    all_sweet_ops = []   # all qualifying ops across every bot

    for i, bid in enumerate(SWEET_SPOT_BOT_IDS, 1):
        name = bot_names.get(bid, str(bid))
        print(f"[{i:>2}/{len(SWEET_SPOT_BOT_IDS)}] bot {bid} — {name[:50]}")
        raw_ops  = fetch_all_ops(bid)
        rows     = extract_features(raw_ops)
        label, sweet_rows = best_sweet_spot(rows)
        if sweet_rows:
            net = sum(r["pl"] for r in sweet_rows)
            print(f"         ✓  {label}  →  {len(sweet_rows)} ops | P&L {net:+.2f}")
            all_sweet_ops.extend(sweet_rows)
        else:
            print(f"         ✗  no sweet spot")

    if not all_sweet_ops:
        print("No sweet-spot ops found.")
        return

    # sort by date
    all_sweet_ops.sort(key=lambda r: r["placedAt"])
    print(f"\nTotal sweet-spot ops: {len(all_sweet_ops)}")

    # ── build cumulative series ───────────────────────────────────────────────
    dates     = [datetime.fromisoformat(r["placedAt"].replace("Z", "+00:00"))
                 for r in all_sweet_ops]
    cum_inv   = np.cumsum([r["exposure"] for r in all_sweet_ops])
    cum_ret   = np.cumsum([r["returned"] for r in all_sweet_ops])
    cum_pl    = np.cumsum([r["pl"]       for r in all_sweet_ops])

    # ── weekly net cash-flow for bar chart ───────────────────────────────────
    week_data = defaultdict(lambda: {"staked": 0.0, "returned": 0.0, "pl": 0.0})
    for r, d in zip(all_sweet_ops, dates):
        # ISO week start (Monday)
        week_start = (d - timedelta(days=d.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0,
            tzinfo=timezone.utc)
        week_data[week_start]["staked"]   += r["exposure"]
        week_data[week_start]["returned"] += r["returned"]
        week_data[week_start]["pl"]       += r["pl"]

    week_dates  = sorted(week_data)
    week_pl     = [week_data[w]["pl"]     for w in week_dates]
    week_staked = [week_data[w]["staked"] for w in week_dates]

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 9),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12}
    )

    # ── top panel: cumulative invested vs returned ────────────────────────────
    ax1.fill_between(dates, 0, cum_inv,
                     alpha=0.25, color="#3498db", label="Cumulative invested (exposure)")
    ax1.fill_between(dates, 0, cum_ret,
                     alpha=0.30, color="#2ecc71", label="Cumulative returned")

    ax1.plot(dates, cum_inv, color="#2980b9", linewidth=1.4, zorder=3)
    ax1.plot(dates, cum_ret, color="#27ae60", linewidth=1.4, zorder=3)

    # net P&L line (white / orange)
    ax1.plot(dates, cum_pl, color="#f39c12", linewidth=2.2,
             zorder=5, label="Cumulative net P&L")
    ax1.axhline(0, color="#aaa", linewidth=0.7, linestyle="--")

    # shade profit vs loss zones on the gap
    ax1.fill_between(dates, cum_pl, 0,
                     where=cum_pl >= 0, alpha=0.15, color="#2ecc71")
    ax1.fill_between(dates, cum_pl, 0,
                     where=cum_pl <  0, alpha=0.15, color="#e74c3c")

    # annotations
    final_pl  = cum_pl[-1]
    final_inv = cum_inv[-1]
    final_ret = cum_ret[-1]
    ax1.annotate(f"Total invested: {final_inv:,.0f}",
                 xy=(dates[-1], cum_inv[-1]), xytext=(-130, 12),
                 textcoords="offset points", fontsize=8.5, color="#2980b9",
                 arrowprops=dict(arrowstyle="->", color="#2980b9", lw=0.8))
    ax1.annotate(f"Total returned: {final_ret:,.0f}",
                 xy=(dates[-1], cum_ret[-1]), xytext=(-130, -18),
                 textcoords="offset points", fontsize=8.5, color="#27ae60",
                 arrowprops=dict(arrowstyle="->", color="#27ae60", lw=0.8))
    ax1.annotate(f"Net P&L: {final_pl:+,.0f}",
                 xy=(dates[-1], cum_pl[-1]), xytext=(-90, 22),
                 textcoords="offset points", fontsize=9.5, color="#e67e22",
                 fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1))

    ax1.set_title(
        f"All 46 Sweet-Spot Bots — Aggregated Exposure & Returns\n"
        f"{len(all_sweet_ops)} sweet-spot ops | "
        f"Total staked: {final_inv:,.0f} | "
        f"Total returned: {final_ret:,.0f} | "
        f"Net P&L: {final_pl:+,.0f}",
        fontsize=11, pad=12)
    ax1.set_ylabel("Cumulative Value (R$)", fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}"))
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.7)
    ax1.grid(True, alpha=0.2, linestyle="--")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── bottom panel: weekly net P&L bars ────────────────────────────────────
    bar_colors = ["#2ecc71" if p >= 0 else "#e74c3c" for p in week_pl]
    ax2.bar(week_dates, week_pl, width=5, color=bar_colors,
            alpha=0.75, align="edge")
    ax2.axhline(0, color="#aaa", linewidth=0.7, linestyle="--")
    ax2.set_ylabel("Weekly P&L", fontsize=9)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.grid(True, alpha=0.15, linestyle="--", axis="y")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:+,.0f}"))
    plt.setp(ax2.get_xticklabels(), rotation=35, ha="right")

    plt.tight_layout()
    out = "/home/user/gui/sweetspot_exposure_graph.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
