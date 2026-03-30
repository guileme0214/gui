"""
Sweet-spot analysis v2 — Risk-Adjusted Day Trade Framework
===========================================================
Key changes from v1
  1. Variable-stake bots excluded:
       MARTINGALE, CYCLES_METHOD, STAKE_OVER_PROFIT_LOSS
       → stake depends on previous result, cumulative P&L is meaningless
  2. LAY risk penalty:
       max_loss_ratio = worst single loss / total P&L
       → one bad LAY at high odds can wipe many wins
  3. New ranking metric (replaces total_pl × r2 × wr):
       score = r2 × sharpe × profit_factor
               / (dd_pct + 0.5 × max_loss_ratio + 0.05)
       where:
         sharpe        = mean(pl) / std(pl)   (consistency per bet)
         profit_factor = sum(wins) / sum(|losses|)  (efficiency)
         max_loss_ratio = |worst loss| / total_pl
  4. Tighter thresholds: MIN_R2 = 0.50, MAX_DD_PCT = 0.50
  5. Chart annotates: strategy (LAY/BACK/Mixed), Sharpe, PF, max loss
"""

import requests, time, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from collections import defaultdict

# ── config ───────────────────────────────────────────────────────────────────
BOTS_URL   = "https://bot.bolsadeaposta.bet.br/api/project/bots"
OPS_URL    = "https://bot.bolsadeaposta.bet.br/api/project/bots/{bot_id}/operations"
COMMISSION = 4.5
PAGE_SIZE  = 50

MIN_OPS    = 8
MIN_R2     = 0.50      # require more linear growth (was 0.45)
MAX_DD_PCT = 0.50      # tighter max drawdown (was 0.55)

# Variable-stake strategies → excluded from analysis
EXCLUDED_STRATEGIES = {"MARTINGALE", "CYCLES_METHOD", "STAKE_OVER_PROFIT_LOSS"}

OUT_DIR = "/home/user/gui/sweetspot_charts_v2"
os.makedirs(OUT_DIR, exist_ok=True)

SESSION = requests.Session()
SESSION.headers["Accept"] = "application/json"


# ── fetch ─────────────────────────────────────────────────────────────────────
def fetch_bots():
    r = SESSION.get(BOTS_URL, timeout=20)
    r.raise_for_status()
    body = r.json()
    raw = body.get("data", body) if isinstance(body, dict) else body
    if isinstance(raw, list):
        bots = raw
    else:
        bots = raw.get("bots", raw.get("items", []))
    return bots


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
            print(f"    ⚠  bot {bot_id} page {page}: {e}")
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


# ── feature extraction ────────────────────────────────────────────────────────
def extract_features(ops):
    rows = []
    for op in ops:
        bets = op.get("bets", [])
        if not bets:
            continue
        b = bets[0]
        placed = op.get("placedAt", "")
        if not placed:
            continue
        rows.append({
            "placedAt":           placed,
            "pl":                 op.get("pl", 0) or 0,
            "side":               b.get("side", "BACK"),
            "matchedOdd":         b.get("matchedOdd"),
            "matchedSize":        b.get("matchedSize") or 0,
            "minute":             b.get("minute"),
            "homeScore":          b.get("homeScore"),
            "awayScore":          b.get("awayScore"),
            "period":             b.get("period"),
            "marketTotalMatched": b.get("marketTotalMatched"),
            "selectionName":      b.get("selectionName", ""),
        })
    rows.sort(key=lambda r: r["placedAt"])
    return rows


# ── bucket helpers ────────────────────────────────────────────────────────────
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


# ── scoring ───────────────────────────────────────────────────────────────────
def score_slice(rows):
    n = len(rows)
    if n < MIN_OPS:
        return {"valid": False}

    pl_vals = [r["pl"] for r in rows]
    cum = np.cumsum(pl_vals)
    total_pl = float(cum[-1])

    if total_pl <= 0:
        return {"valid": False}

    # linear trend
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, cum, 1)
    if slope <= 0:
        return {"valid": False}

    ss_res = np.sum((cum - (slope * x + intercept)) ** 2)
    ss_tot = np.sum((cum - cum.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
    if r2 < MIN_R2:
        return {"valid": False}

    # drawdown
    peak = np.maximum.accumulate(cum)
    dd_pct = float(np.max(peak - cum) / peak[-1]) if peak[-1] > 0 else 1.0
    if dd_pct > MAX_DD_PCT:
        return {"valid": False}

    # risk metrics
    wins   = [p for p in pl_vals if p > 0]
    losses = [p for p in pl_vals if p < 0]
    win_rate     = len(wins) / n
    wins_sum     = sum(wins)
    loss_sum     = abs(sum(losses)) if losses else 0
    profit_factor = wins_sum / (loss_sum + 1e-9)

    # Sharpe per bet
    mean_pl = float(np.mean(pl_vals))
    std_pl  = float(np.std(pl_vals)) + 1e-9
    sharpe  = mean_pl / std_pl

    # LAY risk: worst single loss vs total profit
    max_loss      = abs(min(pl_vals)) if losses else 0.0
    max_loss_ratio = max_loss / total_pl

    # strategy type
    sides = [r["side"] for r in rows]
    lay_pct = sides.count("LAY") / len(sides)
    if lay_pct >= 0.75:
        strategy = "LAY"
    elif lay_pct <= 0.25:
        strategy = "BACK"
    else:
        strategy = "Mixed"

    return {
        "valid":          True,
        "n":              n,
        "total_pl":       total_pl,
        "slope":          float(slope),
        "r2":             r2,
        "dd_pct":         dd_pct,
        "win_rate":       win_rate,
        "wins":           len(wins),
        "losses":         len(losses),
        "profit_factor":  profit_factor,
        "sharpe":         sharpe,
        "max_loss":       max_loss,
        "max_loss_ratio": max_loss_ratio,
        "strategy":       strategy,
        "cum":            cum.tolist(),
        "dates":          [r["placedAt"] for r in rows],
        "pl_vals":        pl_vals,
    }


def rank_score(sc):
    """Risk-adjusted ranking. Higher = better."""
    sharpe_pos = max(sc["sharpe"], 0.01)
    return (
        sc["r2"] * sharpe_pos * sc["profit_factor"]
        / (sc["dd_pct"] + 0.5 * sc["max_loss_ratio"] + 0.05)
    )


# ── sweet-spot detection ──────────────────────────────────────────────────────
def find_sweet_spots(rows):
    results = []

    # single-dimension
    for dim, fn in BUCKET_FUNCS.items():
        groups = defaultdict(list)
        for r in rows:
            v = fn(r)
            if v is not None:
                groups[v].append(r)
        for v, grp in groups.items():
            sc = score_slice(grp)
            if sc["valid"]:
                results.append((f"{dim}={v}", sc))

    # 2-D combos
    for d1, d2 in [("odd","minute"), ("odd","score"), ("odd","liquidity"),
                   ("minute","score"), ("odd","period"), ("minute","liquidity"),
                   ("score","liquidity")]:
        fn1, fn2 = BUCKET_FUNCS[d1], BUCKET_FUNCS[d2]
        groups = defaultdict(list)
        for r in rows:
            v1, v2 = fn1(r), fn2(r)
            if v1 is not None and v2 is not None:
                groups[(v1, v2)].append(r)
        for (v1, v2), grp in groups.items():
            sc = score_slice(grp)
            if sc["valid"]:
                results.append((f"{d1}={v1} & {d2}={v2}", sc))

    if not results:
        return []

    results.sort(key=lambda x: rank_score(x[1]), reverse=True)
    return results


# ── plot ──────────────────────────────────────────────────────────────────────
STRATEGY_COLOR = {"LAY": "#e74c3c", "BACK": "#2980b9", "Mixed": "#8e44ad"}

def plot_sweet_spot(info):
    sc       = info["score"]
    bot_id   = info["bot_id"]
    bot_name = info["bot_name"]
    label    = info["label"]

    dates = [datetime.fromisoformat(d.replace("Z", "+00:00")) for d in sc["dates"]]
    cum   = sc["cum"]

    fig, ax = plt.subplots(figsize=(12, 5))

    # cumulative P&L line (green/red segments)
    for i in range(1, len(dates)):
        color = "#2ecc71" if cum[i] >= cum[i-1] else "#e74c3c"
        ax.plot(dates[i-1:i+1], cum[i-1:i+1], color=color, linewidth=2)

    # trend line
    x_num = np.arange(len(dates), dtype=float)
    slope, intercept = np.polyfit(x_num, cum, 1)
    trend = slope * x_num + intercept
    ax.plot(dates, trend, color="#f39c12", linewidth=1.2,
            linestyle="--", alpha=0.8, label=f"Trend (R²={sc['r2']:.2f})")

    ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")

    # scatter dots
    for i, pl in enumerate(sc["pl_vals"]):
        ax.scatter(dates[i], cum[i],
                   color="#27ae60" if pl >= 0 else "#c0392b",
                   zorder=5, s=35, edgecolors="white", linewidths=0.5)

    # fill
    ax.fill_between(dates, cum, 0,
                    where=[v >= 0 for v in cum], alpha=0.09, color="#2ecc71")
    ax.fill_between(dates, cum, 0,
                    where=[v <  0 for v in cum], alpha=0.09, color="#e74c3c")

    # strategy label
    strat = sc["strategy"]
    strat_color = STRATEGY_COLOR.get(strat, "#555")
    ax.text(0.01, 0.97, strat, transform=ax.transAxes,
            fontsize=9, color=strat_color, fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=strat_color, alpha=0.8))

    # stats box (top-right)
    max_loss_warn = "  ⚠" if sc["max_loss_ratio"] > 0.5 else ""
    stats_text = (
        f"Sharpe: {sc['sharpe']:.2f}\n"
        f"PF: {sc['profit_factor']:.2f}\n"
        f"Max loss: {sc['max_loss']:.0f} ({sc['max_loss_ratio']*100:.0f}% of PL){max_loss_warn}\n"
        f"DD: {sc['dd_pct']*100:.1f}%"
    )
    ax.text(0.99, 0.97, stats_text, transform=ax.transAxes,
            fontsize=7.5, va="top", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bbb", alpha=0.85))

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=35, ha="right")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.01, 0.88))

    title = (
        f"Bot {bot_id} — {bot_name}\n"
        f"Sweet spot: {label}\n"
        f"{sc['n']} ops | {sc['wins']}W / {sc['losses']}L "
        f"({sc['win_rate']*100:.1f}%) | R²={sc['r2']:.2f} | Net P&L: {sc['total_pl']:+.2f}"
    )
    ax.set_title(title, fontsize=9, pad=10)
    ax.set_ylabel("Cumulative P&L", fontsize=9)
    ax.grid(True, alpha=0.22, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    fname = os.path.join(OUT_DIR, f"bot_{bot_id}_sweetspot_v2.png")
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()
    return fname


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("Fetching bots …")
    bots = fetch_bots()
    print(f"Total bots: {len(bots)}\n")

    fixed_bots = [b for b in bots
                  if b.get("stakeManagementName") not in EXCLUDED_STRATEGIES]
    excl_bots  = [b for b in bots
                  if b.get("stakeManagementName") in EXCLUDED_STRATEGIES]

    print(f"Excluded (variable stake): {len(excl_bots)}")
    print(f"Eligible (fixed stake): {len(fixed_bots)}\n")

    sweet_spots = []
    no_spot     = []

    for i, bot in enumerate(fixed_bots, 1):
        bid  = bot["id"]
        name = bot.get("name", str(bid))
        strat_name = bot.get("stakeManagementName", "?")
        print(f"[{i:>3}/{len(fixed_bots)}] {bid} — {name[:50]}  [{strat_name}]")

        ops  = fetch_all_ops(bid)
        rows = extract_features(ops)

        if len(rows) < MIN_OPS:
            no_spot.append({"id": bid, "name": name, "reason": f"only {len(rows)} ops"})
            print(f"         ✗  only {len(rows)} ops")
            continue

        candidates = find_sweet_spots(rows)

        if not candidates:
            no_spot.append({"id": bid, "name": name, "reason": "no valid sweet spot"})
            print(f"         ✗  no valid sweet spot")
            continue

        best_label, best_sc = candidates[0]
        info = {
            "bot_id":      bid,
            "bot_name":    name,
            "label":       best_label,
            "score":       best_sc,
            "all_ops":     len(rows),
            "top5":        candidates[:5],
            "rank_score":  rank_score(best_sc),
        }
        sweet_spots.append(info)
        fname = plot_sweet_spot(info)

        warn = " ⚠ LAY risk" if best_sc["max_loss_ratio"] > 0.5 else ""
        print(
            f"         ✓  {best_label}\n"
            f"            {best_sc['n']} ops | WR {best_sc['win_rate']*100:.1f}% | "
            f"R²={best_sc['r2']:.2f} | Sharpe={best_sc['sharpe']:.2f} | "
            f"PF={best_sc['profit_factor']:.2f} | "
            f"MaxLoss={best_sc['max_loss']:.0f}({best_sc['max_loss_ratio']*100:.0f}%) | "
            f"P&L={best_sc['total_pl']:+.2f}{warn}"
        )

    # ── report ────────────────────────────────────────────────────────────────
    # rank by composite score
    sweet_spots.sort(key=lambda x: x["rank_score"], reverse=True)

    lines = []
    lines.append(f"Sweet-spot analysis v2 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Eligible bots: {len(fixed_bots)}  |  Sweet spots: {len(sweet_spots)}  |  None: {len(no_spot)}")
    lines.append(f"Excluded (variable stake): {len(excl_bots)}")
    lines.append(f"Scoring: R² × Sharpe × ProfitFactor / (DD + 0.5×MaxLossRatio + 0.05)")
    lines.append("")

    for rank_i, info in enumerate(sweet_spots, 1):
        sc = info["score"]
        warn = "  ⚠ LAY RISK: single loss > 50% of P&L" if sc["max_loss_ratio"] > 0.5 else ""
        top5_txt = "\n".join(
            f"      #{j+1}  {lbl}  →  {s['n']} ops | WR {s['win_rate']*100:.1f}% | "
            f"R²={s['r2']:.2f} | Sharpe={s['sharpe']:.2f} | PF={s['profit_factor']:.2f} | "
            f"P&L {s['total_pl']:+.2f}"
            for j, (lbl, s) in enumerate(info["top5"])
        )
        lines.append(f"\n{'─'*70}")
        lines.append(f"#{rank_i}  BOT {info['bot_id']}  {info['bot_name']}")
        lines.append(f"  Strategy: {sc['strategy']}  |  Total ops (bot): {info['all_ops']}")
        lines.append(f"  Best sweet spot: {info['label']}{warn}")
        lines.append(
            f"    Ops: {sc['n']} | Win rate: {sc['win_rate']*100:.1f}% | "
            f"R²: {sc['r2']:.2f} | Sharpe: {sc['sharpe']:.2f} | "
            f"Profit Factor: {sc['profit_factor']:.2f}"
        )
        lines.append(
            f"    DD: {sc['dd_pct']*100:.1f}% | "
            f"Max single loss: {sc['max_loss']:.1f} ({sc['max_loss_ratio']*100:.0f}% of P&L) | "
            f"Net P&L: {sc['total_pl']:+.2f}"
        )
        lines.append(f"  Top-5 candidates:\n{top5_txt}")

    lines.append(f"\n\n{'='*70}")
    lines.append("EXCLUDED — VARIABLE STAKE (Martingale / Cycles / etc.)")
    lines.append("  These bots cannot be reliably analysed with cumulative P&L graphs")
    lines.append("  because the stake size depends on previous bet results.")
    for b in excl_bots:
        lines.append(f"  • [{b.get('stakeManagementName')}] {b['id']} — {b.get('name','')}")

    lines.append(f"\n{'='*70}")
    lines.append("FIXED-STAKE BOTS WITH NO SWEET SPOT FOUND")
    for b in no_spot:
        lines.append(f"  • {b['id']} — {b['name']} ({b['reason']})")

    report = "\n".join(lines)
    path   = "/home/user/gui/sweetspot_report_v2.txt"
    with open(path, "w") as f:
        f.write(report)

    print(f"\n{'='*70}")
    print(f"Sweet spots found: {len(sweet_spots)} / {len(fixed_bots)} eligible bots")
    print(f"Charts → {OUT_DIR}/")
    print(f"Report → {path}")


if __name__ == "__main__":
    main()
