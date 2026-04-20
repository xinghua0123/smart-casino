"""Generate architecture diagram for Smart Casino Floor demo."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(20, 12))
ax.set_xlim(0, 20)
ax.set_ylim(0, 12)
ax.axis("off")
fig.patch.set_facecolor("white")

# --- Colors ---
C_KAFKA   = "#FF6B35"
C_RW_BG   = "#E8F4FD"
C_RW_COMP = "#1A73E8"
C_RW_NEW  = "#0F9D58"   # Theo Win / house-edge highlight color
C_ML      = "#7B2D8E"
C_DASH    = "#0D9488"
C_AGENT   = "#C026D3"
C_PROD    = "#475569"
C_ARROW   = "#64748B"
C_TEXT    = "#1E293B"

def box(x, y, w, h, color, label, fontsize=9, textcolor="white", bold=False):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                       facecolor=color, edgecolor="none", zorder=3)
    ax.add_patch(b)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=fontsize, color=textcolor, weight=weight, zorder=4)

def arrow(x1, y1, x2, y2, label="", color=C_ARROW, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=1.5),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.2, label, ha="center", va="bottom",
                fontsize=7, color=color, style="italic", zorder=4)

# --- Title ---
ax.text(10, 11.5, "Smart Casino Floor — RisingWave + ML + Theoretical Win",
        ha="center", va="center", fontsize=17, weight="bold", color=C_TEXT)
ax.text(10, 11.0,
        "Real-time streaming feature store · High-roller lookalike detection · TheoWin / house-edge economics · LLM chat agent",
        ha="center", va="center", fontsize=9.5, color="#475569", style="italic")

# === Left: Data Sources ===
box(0.3, 8.3, 2.2, 0.7, C_PROD, "Gaming Events\n(slots, roulette,\nblackjack, poker)", 8)
box(0.3, 7.3, 2.2, 0.7, C_PROD, "F&B Events\n(dining, drinks)", 8)
box(0.3, 6.3, 2.2, 0.7, C_PROD, "Hotel Events\n(stays, spa, minibar)", 8)
ax.text(1.4, 9.5, "Data Producer\n(simulated player streams)",
        ha="center", fontsize=8, color=C_PROD, weight="bold")

# === Kafka ===
box(3.5, 7.0, 1.8, 2.2, C_KAFKA, "Apache\nKafka\n\n3 topics", 10, bold=True)
arrow(2.5, 8.6, 3.5, 8.4, "")
arrow(2.5, 7.65, 3.5, 7.9, "")
arrow(2.5, 6.65, 3.5, 7.4, "")

# === RisingWave box (large container) ===
rw_rect = FancyBboxPatch((6, 2.8), 7.2, 7.7, boxstyle="round,pad=0.2",
                          facecolor=C_RW_BG, edgecolor=C_RW_COMP,
                          linewidth=2, zorder=1)
ax.add_patch(rw_rect)
ax.text(9.6, 10.15, "RisingWave (streaming SQL)", ha="center", fontsize=13,
        weight="bold", color=C_RW_COMP)

# --- RW: Sources ---
box(6.3, 8.5, 1.9, 0.8, C_RW_COMP, "Kafka Sources\n(3 streams)", 9)
arrow(5.3, 8.1, 6.3, 8.7, "Ingest")

# --- RW: Feature MVs (gaming/FNB/hotel session) ---
box(6.3, 6.9, 1.9, 1.2, C_RW_COMP,
    "Session\nFeature MVs\n\nTUMBLE 5-min\nper player", 8)
arrow(7.25, 8.5, 7.25, 8.1, "TUMBLE\nwindows", C_RW_COMP)

# --- RW: Theo Win MVs (new, highlighted) ---
box(8.5, 6.9, 2.1, 1.2, C_RW_NEW,
    "Theo Win MVs\n\ntheo_win_window\ncumulative_theo_win\neffective_house_edge", 8, bold=True)
arrow(8.2, 7.5, 8.5, 7.5, "", C_RW_NEW)

# --- RW: HR Similarity MV ---
box(10.9, 6.9, 2.1, 1.2, C_RW_COMP,
    "High Roller\nSimilarity MV\n\nweighted score\n(incl. theo_win 20%)", 8)
arrow(10.6, 7.5, 10.9, 7.5, "", C_RW_COMP)

# --- RW: mv_player_features (combined) ---
box(6.3, 5.4, 6.7, 0.8, C_RW_COMP,
    "mv_player_features — unified feature store\n(gaming × F&B × hotel × theo × house_edge)",
    8, bold=True)
arrow(7.25, 6.9, 7.25, 6.2, "", C_RW_COMP)
arrow(9.55, 6.9, 9.55, 6.2, "", C_RW_COMP)
arrow(11.95, 6.9, 11.95, 6.2, "", C_RW_COMP)

# --- RW: Recommendations Table ---
box(6.3, 3.9, 2.5, 1.0, C_RW_COMP,
    "recommendations_tbl\n(ML predictions)", 8)

# --- RW: Actionable Recommendations MV ---
box(9.1, 3.9, 3.9, 1.2, C_RW_COMP,
    "Actionable Recommendations MV\n\nbusiness rules + offer_value\n(% of cumulative theo_win)",
    8, bold=True)
arrow(8.8, 4.4, 9.1, 4.4, "JOIN", C_RW_COMP)
arrow(11.0, 5.4, 11.0, 5.1, "", C_RW_COMP)

# --- RW: Dashboard MVs ---
box(6.3, 3.0, 6.7, 0.7, C_RW_COMP,
    "Dashboard MVs: mv_theo_by_tier · mv_high_roller_radar · mv_dashboard_stats", 8)
arrow(7.25, 5.4, 7.25, 3.7, "", C_RW_COMP)

# === ML Service ===
ml_rect = FancyBboxPatch((13.8, 5.5), 3.7, 4.5, boxstyle="round,pad=0.2",
                          facecolor="#F3E8FF", edgecolor=C_ML,
                          linewidth=2, zorder=1)
ax.add_patch(ml_rect)
ax.text(15.65, 9.7, "ML Service (Python)", ha="center", fontsize=10.5,
        weight="bold", color=C_ML)

box(14.1, 8.3, 3.1, 0.8, C_ML, "Real-time Features\n(query RW MVs)", 8)
box(14.1, 7.0, 3.1, 1.0, C_ML,
    "scikit-learn Models\n\nnext-game | churn\noffer | HR trajectory", 8)
box(14.1, 5.7, 3.1, 0.8, C_ML, "Predictions\n(every 10s)", 8)

arrow(15.65, 8.3, 15.65, 8.0, "")
arrow(15.65, 7.0, 15.65, 6.5, "Predict")

# RW -> ML (features out)
arrow(13.0, 7.5, 14.1, 8.5, "Query features", C_ML)

# ML -> RW (predictions back via SQL INSERT)
arrow(14.1, 5.9, 8.8, 4.4, "INSERT predictions\n(direct SQL)", C_ML)

# === Dashboard ===
box(7.5, 0.9, 3.5, 0.9, C_DASH, "Streamlit Dashboard\nlocalhost:8501", 10, bold=True)
arrow(9.25, 3.0, 9.25, 1.8, "Query MVs", C_DASH)

# KPI caption under the dashboard
ax.text(9.25, 0.45,
        "Live KPIs: Active Players · Avg Bet · Total Wagered · Theo Win (window) · Effective House Edge",
        ha="center", fontsize=7.5, color=C_DASH, style="italic")

# === LLM Chat Agent ===
agent_rect = FancyBboxPatch((13.8, 0.6), 5.8, 4.6, boxstyle="round,pad=0.2",
                             facecolor="#FDF4FF", edgecolor=C_AGENT,
                             linewidth=2, zorder=1)
ax.add_patch(agent_rect)
ax.text(16.7, 4.95, "LLM Chat Agent", ha="center", fontsize=10.5,
        weight="bold", color=C_AGENT)
ax.text(16.7, 4.55, "(sidebar of the dashboard)", ha="center", fontsize=8,
        style="italic", color=C_AGENT)

box(14.1, 3.4, 5.2, 0.9, C_AGENT,
    "Pluggable providers:\nClaude · OpenAI · OpenRouter · Azure", 8)
box(14.1, 2.3, 5.2, 0.9, C_AGENT,
    "NL → SQL (schema-aware system prompt)\n→ execute on RW → summarize", 8)
box(14.1, 1.2, 5.2, 0.9, C_AGENT,
    "Custom base_url support\n(proxy / gateway compatible)", 8)

# Dashboard <-> Agent
arrow(11.0, 1.35, 13.8, 1.65, "embeds", C_AGENT)
# Agent queries the same RW MVs
arrow(14.1, 2.75, 13.0, 3.35, "SELECT", C_AGENT)

# === House Advantage reference (bottom-left panel) ===
ax.text(0.4, 5.4, "House Advantage (house edge):", fontsize=9,
        weight="bold", color=C_TEXT)
edges = [("Slots", "7.50%"), ("Roulette", "5.26%"),
         ("Blackjack", "0.75%"), ("Poker", "2.50%")]
for i, (game, pct) in enumerate(edges):
    ax.text(0.4, 4.95 - i * 0.35, f"  • {game}: {pct}",
            fontsize=8, color=C_TEXT)

# Theo Win formula call-out
ax.text(0.4, 3.3, "Theo Win formula", fontsize=9, weight="bold", color=C_RW_NEW)
ax.text(0.4, 2.85,
        r"theo_win = $\Sigma$ ( bet × house_edge )",
        fontsize=8.5, color=C_RW_NEW, family="serif")
ax.text(0.4, 2.55, "Casino's expected profit, regardless",
        fontsize=7.5, color=C_RW_NEW, style="italic")
ax.text(0.4, 2.30, "of short-term luck — the industry-",
        fontsize=7.5, color=C_RW_NEW, style="italic")
ax.text(0.4, 2.05, "standard player-value metric.",
        fontsize=7.5, color=C_RW_NEW, style="italic")

# === Legend ===
ax.text(0.4, 1.5, "Components:", fontsize=9, weight="bold", color=C_TEXT)
legend_items = [
    (C_PROD,    "Event producer"),
    (C_KAFKA,   "Message broker"),
    (C_RW_COMP, "RisingWave MV"),
    (C_RW_NEW,  "Theo Win / house-edge"),
    (C_ML,      "ML inference"),
    (C_DASH,    "Dashboard"),
    (C_AGENT,   "LLM chat agent"),
]
for i, (color, label) in enumerate(legend_items):
    y = 1.15 - i * 0.3
    b = FancyBboxPatch((0.4, y - 0.09), 0.35, 0.22, boxstyle="round,pad=0.04",
                       facecolor=color, edgecolor="none", zorder=3)
    ax.add_patch(b)
    ax.text(0.9, y + 0.02, label, fontsize=7.5, va="center", color=C_TEXT)

plt.tight_layout()
plt.savefig("/Users/ronxing/Documents/local/demo/smart-casino-floor/architecture.png",
            dpi=150, bbox_inches="tight", facecolor="white")
print("Saved to architecture.png")
