"""
Sweet-spot analysis for all bots.
For each bot we:
  1. Paginate all operations
  2. Extract 1st-bet features: matchedOdd, minute, homeScore, awayScore,
     period, marketTotalMatched, selectionName
  3. Try every single-dimension filter bucket and every 2-D combo
  4. Score each slice: needs >= MIN_OPS operations, positive total PL,
     positive linear trend (slope & R²), and drawdown within limits
  5. Report the best sweet-spot per bot (if any) and save a PNG for it
"""

import requests, time, json, os, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from itertools import product
from collections import defaultdict

# ── config ──────────────────────────────────────────────────────────────────
BOTS_URL   = "https://bot.bolsadeaposta.bet.br/api/project/bots"
OPS_URL    = "https://bot.bolsadeaposta.bet.br/api/project/bots/{bot_id}/operations"
COMMISSION = 4.5
PAGE_SIZE  = 50
MIN_OPS    = 8          # min operations in a sweet-spot slice
MIN_R2     = 0.45       # min R² on cumulative-PL linear trend
MAX_DD_PCT = 0.55       # max drawdown as fraction of peak  (55%)
OUT_DIR    = "/home/user/gui/sweetspot_charts"
os.makedirs(OUT_DIR, exist_ok=True)

SESSION = requests.Session()
SESSION.headers["Accept"] = "application/json"

# ── helpers ──────────────────────────────────────────────────────────────────
def fetch_all_ops(bot_id: int) -> list:
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
            print(f"    ⚠  bot {bot_id} page {page}: {e}")
            break
        body = r.json()
        data = body.get("data", {})
        items = data.get("operations", [])
        if not items:
            break
        ops.extend(items)
        total_pages = data.get("totalPages", 1)
        if page >= total_pages:
            break
        page += 1
        time.sleep(0.12)   # be polite
    return ops


def first_bet_features(op: dict) -> dict | None:
    bets = op.get("bets", [])
    if not bets:
        return None
    b = bets[0]
    return {
        "matchedOdd":          b.get("matchedOdd"),
        "minute":              b.get("minute"),
        "homeScore":           b.get("homeScore"),
        "awayScore":           b.get("awayScore"),
        "period":              b.get("period"),
        "marketTotalMatched":  b.get("marketTotalMatched"),
        "selectionName":       b.get("selectionName", ""),
        "pl":                  op.get("pl", 0) or 0,
        "placedAt":            op.get("placedAt", ""),
    }


def score_slice(rows: list) -> dict:
    """
    rows: list of dicts with 'pl' and 'placedAt' (already filtered, sorted by date)
    Returns scoring dict; valid=True only if all thresholds are met.
    """
    n = len(rows)
    if n < MIN_OPS:
        return {"valid": False}

    pl_vals = [r["pl"] for r in rows]
    cum = np.cumsum(pl_vals)
    total_pl = float(cum[-1])

    if total_pl <= 0:
        return {"valid": False}

    # linear regression on cumulative PL
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, cum, 1)
    if slope <= 0:
        return {"valid": False}

    ss_res = np.sum((cum - (slope * x + intercept)) ** 2)
    ss_tot = np.sum((cum - cum.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0

    if r2 < MIN_R2:
        return {"valid": False}

    # max drawdown
    peak = np.maximum.accumulate(cum)
    dd = np.max(peak - cum)
    dd_pct = dd / peak[-1] if peak[-1] > 0 else 1.0

    if dd_pct > MAX_DD_PCT:
        return {"valid": False}

    wins  = sum(1 for p in pl_vals if p > 0)
    losses = n - wins

    return {
        "valid":    True,
        "n":        n,
        "total_pl": total_pl,
        "slope":    slope,
        "r2":       r2,
        "dd":       float(dd),
        "dd_pct":   dd_pct,
        "win_rate": wins / n,
        "wins":     wins,
        "losses":   losses,
        "cum":      cum.tolist(),
        "dates":    [r["placedAt"] for r in rows],
    }


def bucket_odd(v):
    if v is None: return None
    if v < 2:    return "odd<2"
    if v < 3:    return "2≤odd<3"
    if v < 5:    return "3≤odd<5"
    if v < 8:    return "5≤odd<8"
    if v < 12:   return "8≤odd<12"
    if v < 18:   return "12≤odd<18"
    if v < 25:   return "18≤odd<25"
    if v < 40:   return "25≤odd<40"
    return "odd≥40"

def bucket_minute(v):
    if v is None: return None
    if v < 0:    return "pre-match"
    if v < 15:   return "0-14'"
    if v < 30:   return "15-29'"
    if v < 45:   return "30-44'"
    if v < 60:   return "45-59'"
    if v < 75:   return "60-74'"
    if v < 90:   return "75-89'"
    return "90+'"

def bucket_score(h, a):
    if h is None or a is None: return None
    if h == 0 and a == 0: return "0-0"
    if h > a:  return "home_winning"
    if a > h:  return "away_winning"
    return "level>0"

def bucket_liquidity(v):
    if v is None: return None
    if v < 10_000:   return "liq<10k"
    if v < 50_000:   return "10k≤liq<50k"
    if v < 200_000:  return "50k≤liq<200k"
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

# ── per-bot analysis ─────────────────────────────────────────────────────────
def analyse_bot(bot_id: int, bot_name: str) -> dict | None:
    ops = fetch_all_ops(bot_id)
    if not ops:
        return None

    features = [first_bet_features(op) for op in ops]
    features = [f for f in features if f is not None]
    if len(features) < MIN_OPS:
        return None

    # sort by date (should already be, but just in case)
    features.sort(key=lambda f: f["placedAt"])

    results = []   # list of (label, score_dict)

    # ── single-dimension buckets ──
    for dim, fn in BUCKET_FUNCS.items():
        groups = defaultdict(list)
        for f in features:
            val = fn(f)
            if val is not None:
                groups[val].append(f)
        for val, rows in groups.items():
            sc = score_slice(rows)
            if sc["valid"]:
                results.append((f"{dim}={val}", sc))

    # ── 2-D combos (odd × minute, odd × score, odd × liquidity, minute × score) ──
    combos = [
        ("odd", "minute"),
        ("odd", "score"),
        ("odd", "liquidity"),
        ("minute", "score"),
        ("odd", "period"),
    ]
    for d1, d2 in combos:
        fn1, fn2 = BUCKET_FUNCS[d1], BUCKET_FUNCS[d2]
        groups = defaultdict(list)
        for f in features:
            v1, v2 = fn1(f), fn2(f)
            if v1 is not None and v2 is not None:
                groups[(v1, v2)].append(f)
        for (v1, v2), rows in groups.items():
            sc = score_slice(rows)
            if sc["valid"]:
                results.append((f"{d1}={v1} & {d2}={v2}", sc))

    if not results:
        return None

    # pick the best sweet-spot: rank by composite score
    def rank(item):
        _, sc = item
        # reward total_pl, r2, win_rate; penalise drawdown
        return sc["total_pl"] * sc["r2"] * sc["win_rate"] / (sc["dd_pct"] + 0.01)

    results.sort(key=rank, reverse=True)
    best_label, best_sc = results[0]

    return {
        "bot_id":   bot_id,
        "bot_name": bot_name,
        "label":    best_label,
        "score":    best_sc,
        "all_ops":  len(features),
        "all_results": results[:5],   # top-5 candidates
    }


def plot_sweet_spot(info: dict):
    sc = info["score"]
    dates = [datetime.fromisoformat(d.replace("Z", "+00:00")) for d in sc["dates"]]
    cum   = sc["cum"]
    bot_id   = info["bot_id"]
    bot_name = info["bot_name"]
    label    = info["label"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(1, len(dates)):
        color = "#2ecc71" if cum[i] >= cum[i-1] else "#e74c3c"
        ax.plot(dates[i-1:i+1], cum[i-1:i+1], color=color, linewidth=2)

    ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    for i, (d, c, f) in enumerate(zip(dates, cum,
                                       [{"pl": cum[0]}] +
                                       [{"pl": cum[j]-cum[j-1]} for j in range(1, len(cum))])):
        col = "#27ae60" if (cum[i] - (cum[i-1] if i else 0)) >= 0 else "#c0392b"
        ax.scatter(d, c, color=col, zorder=5, s=35, edgecolors="white", linewidths=0.5)

    ax.fill_between(dates, cum, 0,
        where=[v >= 0 for v in cum], alpha=0.09, color="#2ecc71")
    ax.fill_between(dates, cum, 0,
        where=[v < 0 for v in cum], alpha=0.09, color="#e74c3c")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=35, ha="right")

    title = (f"Bot {bot_id} — {bot_name}\n"
             f"Sweet spot: {label}\n"
             f"{sc['n']} ops | {sc['wins']}W/{sc['losses']}L "
             f"({sc['win_rate']*100:.1f}%) | R²={sc['r2']:.2f} | "
             f"DD={sc['dd_pct']*100:.1f}% | Net: {sc['total_pl']:+.2f}")
    ax.set_title(title, fontsize=9, pad=10)
    ax.set_ylabel("Cumulative P&L", fontsize=9)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    fname = os.path.join(OUT_DIR, f"bot_{bot_id}_sweetspot.png")
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()
    return fname


# ── main ─────────────────────────────────────────────────────────────────────
def fetch_bots() -> list:
    r = SESSION.get(BOTS_URL, timeout=20)
    r.raise_for_status()
    body = r.json()
    # handle both {"data": [...]} and plain list
    raw = body.get("data", body) if isinstance(body, dict) else body
    if isinstance(raw, list):
        return [{"id": b.get("id"), "name": b.get("name", str(b.get("id")))} for b in raw]
    # sometimes nested further
    bots = raw.get("bots", raw.get("items", []))
    return [{"id": b.get("id"), "name": b.get("name", str(b.get("id")))} for b in bots]


def main():
    print("Fetching bot list …")
    bots = fetch_bots()
    print(f"Found {len(bots)} bots\n")

    sweet_spots = []
    no_spot     = []

    for i, bot in enumerate(bots, 1):
        bid  = bot["id"]
        name = bot["name"]
        print(f"[{i:>3}/{len(bots)}] {bid} — {name[:55]}")
        info = analyse_bot(bid, name)
        if info:
            sweet_spots.append(info)
            fname = plot_sweet_spot(info)
            sc = info["score"]
            print(f"         ✓  sweet spot: {info['label']}")
            print(f"            {sc['n']} ops | WR {sc['win_rate']*100:.1f}% | "
                  f"R²={sc['r2']:.2f} | DD {sc['dd_pct']*100:.1f}% | "
                  f"P&L {sc['total_pl']:+.2f} → {fname}")
        else:
            no_spot.append(f"{bid} — {name}")
            print(f"         ✗  no valid sweet spot found")

    # ── summary report ──────────────────────────────────────────────────────
    print("\n" + "="*80)
    print(f"SUMMARY  |  sweet spots found: {len(sweet_spots)}/{len(bots)}")
    print("="*80)

    # rank by total P&L of sweet spot
    sweet_spots.sort(key=lambda x: x["score"]["total_pl"], reverse=True)

    report_lines = []
    for info in sweet_spots:
        sc  = info["score"]
        top5 = "\n".join(
            f"      #{j+1}  {lbl}  →  {s['n']} ops | "
            f"WR {s['win_rate']*100:.1f}% | R²={s['r2']:.2f} | "
            f"P&L {s['total_pl']:+.2f}"
            for j, (lbl, s) in enumerate(info["all_results"])
        )
        line = (
            f"\n{'─'*70}\n"
            f"BOT {info['bot_id']}  {info['bot_name']}\n"
            f"  Total ops: {info['all_ops']}\n"
            f"  Best sweet spot: {info['label']}\n"
            f"    Ops: {sc['n']} | Win rate: {sc['win_rate']*100:.1f}% | "
            f"R²: {sc['r2']:.2f} | Drawdown: {sc['dd_pct']*100:.1f}% | "
            f"Net P&L: {sc['total_pl']:+.2f}\n"
            f"  Top-5 candidates:\n{top5}"
        )
        report_lines.append(line)
        print(line)

    if no_spot:
        print(f"\n{'─'*70}")
        print("BOTS WITH NO SWEET SPOT:")
        for b in no_spot:
            print(f"  • {b}")

    report_path = "/home/user/gui/sweetspot_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Sweet-spot analysis — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Bots analysed: {len(bots)}  |  Sweet spots found: {len(sweet_spots)}\n")
        f.write("Criteria: ≥8 ops, positive P&L, slope>0, R²≥0.45, DD≤55%\n\n")
        f.write("\n".join(report_lines))
        f.write(f"\n\nBOTS WITH NO SWEET SPOT:\n")
        for b in no_spot:
            f.write(f"  • {b}\n")

    print(f"\nReport saved to {report_path}")
    print(f"Charts saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
