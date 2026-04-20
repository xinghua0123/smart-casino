"""Generate architecture diagram for Smart Casino Floor demo."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(18, 11))
ax.set_xlim(0, 18)
ax.set_ylim(0, 11)
ax.axis("off")
fig.patch.set_facecolor("white")

# --- Colors ---
C_KAFKA   = "#FF6B35"
C_RW_BG   = "#E8F4FD"
C_RW_COMP = "#1A73E8"
C_ML      = "#7B2D8E"
C_DASH    = "#0D9488"
C_PROD    = "#475569"
C_ARROW   = "#64748B"
C_TEXT    = "#1E293B"
C_WHITE   = "#FFFFFF"

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
ax.text(9, 10.5, "The Architecture: Smart Casino Floor with RisingWave + ML",
        ha="center", va="center", fontsize=16, weight="bold", color=C_TEXT)

# === Left: Data Sources ===
box(0.3, 7.8, 2.2, 0.7, C_PROD, "Gaming Events\n(slots, table, poker)", 8)
box(0.3, 6.8, 2.2, 0.7, C_PROD, "F&B Events\n(dining, drinks)", 8)
box(0.3, 5.8, 2.2, 0.7, C_PROD, "Hotel Events\n(stays, spa, minibar)", 8)
ax.text(1.4, 9.0, "Data Producer\n(200 simulated players)", ha="center",
        fontsize=8, color=C_PROD, weight="bold")

# === Kafka ===
box(3.5, 6.5, 1.8, 2.2, C_KAFKA, "Apache\nKafka\n\n3 topics", 10, bold=True)
arrow(2.5, 8.1, 3.5, 7.9, "")
arrow(2.5, 7.15, 3.5, 7.3, "")
arrow(2.5, 6.15, 3.5, 6.8, "")

# === RisingWave box (large container) ===
rw_rect = FancyBboxPatch((6, 2.8), 6.5, 7, boxstyle="round,pad=0.2",
                          facecolor=C_RW_BG, edgecolor=C_RW_COMP,
                          linewidth=2, linestyle="-", zorder=1)
ax.add_patch(rw_rect)
ax.text(9.25, 9.5, "RisingWave", ha="center", fontsize=13,
        weight="bold", color=C_RW_COMP)

# --- RW: Sources ---
box(6.5, 7.8, 2.0, 0.8, C_RW_COMP, "Kafka Sources\n(3 streams)", 9)
arrow(5.3, 7.6, 6.5, 8.0, "Ingest")

# --- RW: Feature MVs ---
box(6.5, 6.2, 2.5, 1.2, C_RW_COMP,
    "Feature MVs\n\nsession stats\ngame preferences\ncross-category profiles", 8)
arrow(7.5, 7.8, 7.75, 7.4, "TUMBLE\nwindows", C_RW_COMP)

# --- RW: High Roller Similarity ---
box(9.5, 6.2, 2.5, 1.2, C_RW_COMP,
    "High Roller\nSimilarity MV\n\nweighted score\nvs VIP profile", 8)
arrow(9.0, 6.8, 9.5, 6.8, "", C_RW_COMP)

# --- RW: Recommendations Table ---
box(6.5, 4.2, 2.5, 1.0, C_RW_COMP,
    "recommendations_tbl\n(ML predictions)", 8)

# --- RW: Actionable Recommendations MV ---
box(9.5, 4.2, 2.5, 1.2, C_RW_COMP,
    "Actionable\nRecommendations MV\n\nbusiness rules\noffer calculation", 8)
arrow(9.0, 4.7, 9.5, 4.7, "JOIN", C_RW_COMP)
arrow(10.75, 6.2, 10.75, 5.4, "", C_RW_COMP)

# --- RW: Dashboard Stats ---
box(7.5, 3.0, 3.5, 0.7, C_RW_COMP,
    "Dashboard MVs (stats, radar, distributions)", 8)
arrow(7.75, 6.2, 7.75, 3.7, "", C_RW_COMP)

# === ML Service (right side) ===
ml_rect = FancyBboxPatch((13.3, 4.5), 4.2, 5.0, boxstyle="round,pad=0.2",
                          facecolor="#F3E8FF", edgecolor=C_ML,
                          linewidth=2, zorder=1)
ax.add_patch(ml_rect)
ax.text(15.4, 9.2, "ML Service (Python)", ha="center", fontsize=11,
        weight="bold", color=C_ML)

box(13.8, 7.8, 3.2, 0.8, C_ML, "Real-time Features\n(from RisingWave MVs)", 8)
box(13.8, 6.4, 3.2, 1.1, C_ML,
    "scikit-learn Models\n\nnext-game | churn\noffer | HR trajectory", 8)
box(13.8, 5.0, 3.2, 0.8, C_ML, "Predictions\n(per player, every 10s)", 8)

# ML internal arrows
arrow(15.4, 7.8, 15.4, 7.5, "")
arrow(15.4, 6.4, 15.4, 5.8, "Predict")

# RW -> ML (features out)
arrow(12.0, 7.0, 13.8, 8.0, "Query\nfeatures", C_ML, "->")
arrow(12.5, 6.8, 13.8, 7.0, "", C_ML, "->")

# ML -> RW (predictions back)
arrow(13.8, 5.2, 9.0, 4.7, "INSERT\npredictions", C_ML, "->")

# === Dashboard (bottom) ===
box(7.5, 0.8, 3.5, 1.0, C_DASH, "Streamlit Dashboard\nlocalhost:8501", 10, bold=True)
arrow(9.25, 3.0, 9.25, 1.8, "Query MVs", C_DASH)

# === Legend ===
ax.text(0.5, 2.5, "Data Flow:", fontsize=9, weight="bold", color=C_TEXT)
legend_items = [
    (C_PROD, "Event Sources"),
    (C_KAFKA, "Message Broker"),
    (C_RW_COMP, "RisingWave (streaming SQL)"),
    (C_ML, "ML Inference"),
    (C_DASH, "Visualization"),
]
for i, (color, label) in enumerate(legend_items):
    y = 2.0 - i * 0.4
    b = FancyBboxPatch((0.5, y - 0.1), 0.4, 0.25, boxstyle="round,pad=0.05",
                       facecolor=color, edgecolor="none", zorder=3)
    ax.add_patch(b)
    ax.text(1.1, y + 0.02, label, fontsize=8, va="center", color=C_TEXT)

# --- Key insight annotation ---
ax.text(14.0, 4.0, "Closed-loop: predictions flow\nback into RisingWave for\nbusiness rule evaluation",
        fontsize=7.5, color=C_ML, style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3E8FF", edgecolor=C_ML, alpha=0.7))

plt.tight_layout()
plt.savefig("/Users/ronxing/Documents/local/demo/smart-casino-floor/architecture.png",
            dpi=150, bbox_inches="tight", facecolor="white")
print("Saved to architecture.png")
