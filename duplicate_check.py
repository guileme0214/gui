"""
Check for duplicate/overlapping bets across sweet-spot bots.
Two bots are considered duplicates if they bet on:
  - same marketId  (exact same market)
  - same selectionName (exact same outcome)
  - same side (BACK/LAY)
  - within 5 minutes of each other
"""

import requests, time, json
from datetime import datetime, timezone
from collections import defaultdict

BOTS_URL = "https://bot.bolsadeaposta.bet.br/api/project/bots"
OPS_URL  = "https://bot.bolsadeaposta.bet.br/api/project/bots/{bot_id}/operations"
COMMISSION = 4.5
PAGE_SIZE  = 50

# 46 bots with valid sweet spots
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


def fetch_all_ops(bot_id, bot_name):
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


def parse_op(op, bot_id, bot_name):
    bets = op.get("bets", [])
    if not bets:
        return None
    b = bets[0]
    placed = op.get("placedAt", "")
    try:
        dt = datetime.fromisoformat(placed.replace("Z", "+00:00"))
    except:
        return None
    return {
        "bot_id":        bot_id,
        "bot_name":      bot_name,
        "marketId":      op.get("marketId", ""),
        "marketName":    op.get("marketName", ""),
        "eventName":     op.get("eventName", ""),
        "competitionName": op.get("competitionName", ""),
        "selectionName": b.get("selectionName", ""),
        "side":          b.get("side", ""),
        "matchedOdd":    b.get("matchedOdd"),
        "pl":            op.get("pl", 0),
        "placedAt":      dt,
        "placedAt_str":  placed,
    }


def main():
    # fetch bot names
    r = SESSION.get(BOTS_URL, timeout=20)
    body = r.json()
    raw = body.get("data", body) if isinstance(body, dict) else body
    if isinstance(raw, list):
        bot_names = {b["id"]: b["name"] for b in raw if "id" in b}
    else:
        bots = raw.get("bots", raw.get("items", []))
        bot_names = {b["id"]: b["name"] for b in bots}

    # collect all ops
    all_ops = []
    for i, bid in enumerate(SWEET_SPOT_BOT_IDS, 1):
        name = bot_names.get(bid, str(bid))
        print(f"[{i:>2}/{len(SWEET_SPOT_BOT_IDS)}] fetching bot {bid} — {name[:50]}")
        ops = fetch_all_ops(bid, name)
        parsed = [parse_op(op, bid, name) for op in ops]
        parsed = [p for p in parsed if p]
        all_ops.extend(parsed)
        print(f"       {len(parsed)} ops")

    print(f"\nTotal ops across all sweet-spot bots: {len(all_ops)}")

    # --- Group by marketId + selectionName + side ---
    # Key = (marketId, selectionName, side)
    groups = defaultdict(list)
    for op in all_ops:
        key = (op["marketId"], op["selectionName"], op["side"])
        groups[key].append(op)

    # Find groups where >1 bot placed a bet
    duplicates = []
    for key, ops in groups.items():
        bots_in_group = list({o["bot_id"] for o in ops})
        if len(bots_in_group) < 2:
            continue
        # Check time proximity: at least one pair within 5 min
        ops_sorted = sorted(ops, key=lambda x: x["placedAt"])
        pairs_close = []
        for i in range(len(ops_sorted)):
            for j in range(i+1, len(ops_sorted)):
                a, b = ops_sorted[i], ops_sorted[j]
                if a["bot_id"] == b["bot_id"]:
                    continue
                delta = abs((a["placedAt"] - b["placedAt"]).total_seconds())
                if delta <= 300:  # 5 minutes
                    pairs_close.append((a, b, delta))

        if pairs_close:
            duplicates.append({
                "key":        key,
                "event":      ops[0]["eventName"],
                "competition":ops[0]["competitionName"],
                "market":     ops[0]["marketName"],
                "selection":  key[1],
                "side":       key[2],
                "bots":       {o["bot_id"]: o["bot_name"] for o in ops},
                "ops":        ops_sorted,
                "close_pairs":pairs_close,
                "n_pairs":    len(pairs_close),
            })

    # --- Summarise by bot-pair frequency ---
    pair_counts = defaultdict(list)  # (bid1, bid2) -> list of events
    for dup in duplicates:
        for a, b, delta in dup["close_pairs"]:
            pair = tuple(sorted([a["bot_id"], b["bot_id"]]))
            pair_counts[pair].append({
                "event":     dup["event"],
                "market":    dup["market"],
                "selection": dup["selection"],
                "side":      dup["side"],
                "delta_s":   delta,
                "placedAt":  a["placedAt_str"],
            })

    print(f"\nDuplicate bet events (same market/selection/side within 5 min): {len(duplicates)}")
    print(f"Unique bot pairs involved: {len(pair_counts)}\n")

    # Sort pairs by frequency
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: len(x[1]), reverse=True)

    report_lines = ["=" * 70]
    report_lines.append("DUPLICATE BET ANALYSIS — Sweet-Spot Bots")
    report_lines.append("=" * 70)

    for (bid1, bid2), events in sorted_pairs:
        n = len(events)
        name1 = bot_names.get(bid1, str(bid1))
        name2 = bot_names.get(bid2, str(bid2))
        freq = "FREQUENT" if n >= 5 else ("OCCASIONAL" if n >= 2 else "RARE")
        report_lines.append(f"\n{'─'*70}")
        report_lines.append(f"[{freq}] Bot {bid1} × Bot {bid2}  →  {n} overlapping bet(s)")
        report_lines.append(f"  {name1}")
        report_lines.append(f"  {name2}")
        report_lines.append(f"  Overlaps:")
        for ev in events[:10]:  # show up to 10
            report_lines.append(
                f"    • {ev['placedAt'][:10]}  {ev['event']}"
                f"  [{ev['market']}] {ev['side']} {ev['selection']}"
                f"  (Δ{ev['delta_s']:.0f}s)"
            )
        if len(events) > 10:
            report_lines.append(f"    ... and {len(events)-10} more")

    if not sorted_pairs:
        report_lines.append("\n✓ No duplicate bets found across sweet-spot bots.")

    report = "\n".join(report_lines)
    print(report)

    path = "/home/user/gui/duplicate_report.txt"
    with open(path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {path}")

    # Also save structured JSON for further analysis
    summary = []
    for (bid1, bid2), events in sorted_pairs:
        summary.append({
            "bot1_id": bid1, "bot1_name": bot_names.get(bid1, str(bid1)),
            "bot2_id": bid2, "bot2_name": bot_names.get(bid2, str(bid2)),
            "overlap_count": len(events),
            "frequency": "frequent" if len(events) >= 5 else ("occasional" if len(events) >= 2 else "rare"),
            "events": events[:20],
        })
    with open("/home/user/gui/duplicate_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("JSON saved to /home/user/gui/duplicate_summary.json")


if __name__ == "__main__":
    main()
