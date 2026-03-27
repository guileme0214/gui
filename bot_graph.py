import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

BASE_URL = "https://bot.bolsadeaposta.bet.br/api/project/bots/17898/operations"
PARAMS = {
    "pageSize": 20,
    "sortBy": "placedAt",
    "sortOrder": "asc",
    "commission": 4.5,
}

all_ops = []
page = 1
while True:
    PARAMS["page"] = page
    r = requests.get(BASE_URL, params=PARAMS, timeout=15)
    r.raise_for_status()
    body = r.json()
    data = body["data"]
    items = data["operations"]
    if not items:
        break
    all_ops.extend(items)
    total_pages = data["totalPages"]
    print(f"  fetched page {page}/{total_pages} ({len(items)} ops)")
    if page >= total_pages:
        break
    page += 1

print(f"\nTotal ops fetched: {len(all_ops)}")

# Filter: first bet's matchedOdd between 15 and 30 (inclusive)
filtered = []
for op in all_ops:
    bets = op.get("bets", [])
    if not bets:
        continue
    first_odd = bets[0].get("matchedOdd", 0)
    if 15 <= first_odd <= 30:
        filtered.append(op)

print(f"Ops with first matchedOdd in [15, 30]: {len(filtered)}")

# Build cumulative pl series
dates = []
cum_pl = []
running = 0.0
for op in filtered:
    placed_at = op.get("placedAt", "")
    pl = op.get("pl", 0) or 0
    dt = datetime.fromisoformat(placed_at.replace("Z", "+00:00"))
    running += pl
    dates.append(dt)
    cum_pl.append(running)

# Print summary
print(f"\nDate range: {dates[0].date()} → {dates[-1].date()}")
print(f"Final cumulative P&L: {cum_pl[-1]:.2f}")
wins = sum(1 for op in filtered if (op.get("pl") or 0) > 0)
losses = sum(1 for op in filtered if (op.get("pl") or 0) < 0)
print(f"Wins: {wins} | Losses: {losses} | Win rate: {wins/len(filtered)*100:.1f}%")

# --- Plot ---
fig, ax = plt.subplots(figsize=(13, 6))

# Color line green/red per segment
for i in range(1, len(dates)):
    color = "#2ecc71" if cum_pl[i] >= cum_pl[i - 1] else "#e74c3c"
    ax.plot(dates[i - 1:i + 1], cum_pl[i - 1:i + 1], color=color, linewidth=2)

# Zero line
ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")

# Scatter: green dots = win, red dots = loss
for i, op in enumerate(filtered):
    pl = op.get("pl") or 0
    color = "#27ae60" if pl > 0 else "#c0392b"
    ax.scatter(dates[i], cum_pl[i], color=color, zorder=5, s=40, edgecolors="white", linewidths=0.5)

# Fill under curve
ax.fill_between(dates, cum_pl, 0,
                where=[v >= 0 for v in cum_pl], alpha=0.08, color="#2ecc71")
ax.fill_between(dates, cum_pl, 0,
                where=[v < 0 for v in cum_pl], alpha=0.08, color="#e74c3c")

# Annotations
ax.annotate(f"Peak: {max(cum_pl):.1f}",
            xy=(dates[cum_pl.index(max(cum_pl))], max(cum_pl)),
            xytext=(10, 10), textcoords="offset points",
            fontsize=8.5, color="#27ae60",
            arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1))

ax.annotate(f"Final: {cum_pl[-1]:.1f}",
            xy=(dates[-1], cum_pl[-1]),
            xytext=(-60, 10), textcoords="offset points",
            fontsize=8.5, color="#2c3e50",
            arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1))

ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.xticks(rotation=35, ha="right")

ax.set_title(f"Bot 17898 — Lay 1-0 | Cumulative P&L\n"
             f"Filter: first bet matchedOdd ∈ [15, 30] | "
             f"{len(filtered)} ops | {wins}W / {losses}L ({wins/len(filtered)*100:.1f}%) | "
             f"Net: {cum_pl[-1]:+.2f}",
             fontsize=11, pad=14)
ax.set_xlabel("Date", fontsize=10)
ax.set_ylabel("Cumulative P&L", fontsize=10)
ax.grid(True, alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out = "/home/user/gui/bot17898_cumpl_odds15_30.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved to: {out}")
plt.close()
