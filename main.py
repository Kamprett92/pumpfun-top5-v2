#!/usr/bin/env python3
"""
PumpFun Top-5 Forecast - Improved over house persistence baseline.

Strategy:
- House baseline bets the last snapshot's order holds (persistence).
- We improve by analyzing momentum: coins with strong recent growth
  are more likely to rise; coins fading are more likely to fall.
- We also adjust size predictions based on observed growth trends.

Output format (from starter/main.ts):
{
  "probabilities": [{"mint": str, "place": int, "p": float}],
  "top5": [mint, ...],
  "sizes": [{"place": int, "bin": int, "p": float}]
}
- places: 1-5
- sizes: 82 bins (bin 0 = below edge[0]=2.0, bin 82 = above edge[81]=6.0)
"""
import json
import math
import os
from collections import defaultdict

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

MICRO = 1_000_000

def erfc(x):
    """Complementary error function (Numerical Recipes)."""
    z = abs(x)
    t = 1 / (1 + 0.5 * z)
    r = t * math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return r if x >= 0 else 2 - r

def above(x, mean, sd):
    """P(X >= x) for X ~ Normal(mean, sd)."""
    if x == float('-inf'):
        return 1.0
    if x == float('inf'):
        return 0.0
    return 0.5 * erfc((x - mean) / (sd * math.sqrt(2)))

def size_of(lamports):
    """log10 of cap in SOL: log10(lamports) - 9."""
    return math.log10(float(lamports)) - 9

def analyze_momentum(marks):
    """Track each coin's rank and cap trajectory; compute momentum."""
    coin_history = defaultdict(lambda: {'ranks': [], 'caps': []})
    for mark in marks:
        rows = mark.get("rows", [])
        for row in rows:
            mint = row["mint"]
            coin_history[mint]['ranks'].append(row["rank"])
            coin_history[mint]['caps'].append(float(row["market_cap_lamports"]))
    result = {}
    for mint, h in coin_history.items():
        caps = h['caps']
        ranks = h['ranks']
        if len(caps) < 2:
            result[mint] = {'momentum': 0, 'latest_cap': caps[-1] if caps else 0, 'latest_rank': ranks[-1] if ranks else 999}
            continue
        # Growth: log return from first to last
        if caps[0] > 0 and caps[-1] > 0:
            growth = math.log(caps[-1] / caps[0])
        else:
            growth = 0
        # Rank improvement (positive = moved up)
        rank_change = ranks[0] - ranks[-1] if len(ranks) >= 2 else 0
        momentum = growth * 0.7 + (rank_change / 10.0) * 0.3
        result[mint] = {'momentum': momentum, 'latest_cap': caps[-1], 'latest_rank': ranks[-1]}
    return result

def main():
    task = json.load(open(os.path.join(INPUT_DIR, "task.json")))
    lines = open(os.path.join(INPUT_DIR, "history.ndjson")).read().split("\n")
    lines = [l for l in lines if l.strip()]
    marks = []
    for line in lines:
        try:
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list):
                marks.append(m)
        except:
            pass
    if not marks:
        raise Exception("the history holds no ok mark")
    chart = sorted(marks[-1]["rows"], key=lambda r: r["rank"])
    if len(chart) < task["places"]:
        raise Exception(f"the last mark ranks {len(chart)} coins, fewer than the {task['places']} places")
    momentum = analyze_momentum(marks)
    # Build predicted ranking: adjust last chart by momentum
    scored = []
    for row in chart[:30]:
        mint = row["mint"]
        m = momentum.get(mint, {'momentum': 0})
        log_cap = size_of(row["market_cap_lamports"])
        score = log_cap + m['momentum'] * 0.5
        scored.append((mint, score, float(row["market_cap_lamports"]), row["rank"]))
    scored.sort(key=lambda x: x[1], reverse=True)
    predicted_order = [s[0] for s in scored]
    # Consider high-momentum coins outside top 30
    all_mom = sorted(momentum.items(), key=lambda x: x[1]['momentum'], reverse=True)
    for mint, m in all_mom[:10]:
        if mint not in predicted_order and m['momentum'] > 1.0 and m['latest_cap'] > 0:
            log_cap = math.log10(m['latest_cap']) - 9
            score = log_cap + m['momentum'] * 0.5
            for i, (pm, ps, _, _) in enumerate(scored):
                if score > ps:
                    predicted_order.insert(i, mint)
                    break
    probabilities = []
    edges = task["size"]["edges"]
    top5 = predicted_order[:task["places"]]
    for place in range(1, task["places"] + 1):
        if place - 1 < len(predicted_order):
            pred_mint = predicted_order[place - 1]
            probabilities.append({"mint": pred_mint, "place": place, "p": 0.35})
        if place - 2 >= 0 and place - 2 < len(predicted_order):
            probabilities.append({"mint": predicted_order[place - 2], "place": place, "p": 0.10})
        if place < len(predicted_order):
            probabilities.append({"mint": predicted_order[place], "place": place, "p": 0.10})
        if place - 1 < len(chart):
            chart_mint = chart[place - 1]["mint"]
            if chart_mint not in [p["mint"] for p in probabilities if p["place"] == place]:
                probabilities.append({"mint": chart_mint, "place": place, "p": 0.08})
    sizes = []
    for place in range(1, task["places"] + 1):
        if place - 1 < len(predicted_order):
            pred_mint = predicted_order[place - 1]
            m = momentum.get(pred_mint, {})
            cap = None
            for row in chart:
                if row["mint"] == pred_mint:
                    cap = float(row["market_cap_lamports"])
                    break
            if cap is None:
                cap = m.get('latest_cap', 1e9)
            mom = m.get('momentum', 0)
            growth_factor = math.exp(mom * 2)
            growth_factor = max(0.5, min(growth_factor, 3.0))
            projected_cap = cap * growth_factor
        else:
            projected_cap = 1e9
        mean = size_of(projected_cap)
        sd = 0.40
        for b in range(len(edges) + 1):
            lo = float('-inf') if b == 0 else edges[b - 1]
            hi = float('inf') if b == len(edges) else edges[b]
            u = math.floor((above(lo, mean, sd) - above(hi, mean, sd)) * MICRO)
            if u > 0:
                sizes.append({"place": place, "bin": b, "p": u / MICRO})
    output = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task["output"]), "w") as f:
        f.write(json.dumps(output) + "\n")
    print(f"Forecast written. Top5: {top5}")

if __name__ == "__main__":
    main()
