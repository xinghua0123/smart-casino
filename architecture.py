"""Render the implemented 2.0 architecture as PNG and SVG (no services needed).

Run: python3 architecture.py
Dependency: matplotlib. Outputs are relative to this file, not the caller's cwd.
"""
from pathlib import Path
import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "casino-diagram-cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "casino-diagram-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parent
BG = "#0c1423"
PANEL = "#142236"
TEXT = "#e8f0fa"
MUTED = "#a9bad0"
BLUE = "#82b9f7"
TEAL = "#6dd9c0"
GOLD = "#f2bf6b"
PURPLE = "#c4a0ef"
BORDER = "#30445f"


def render():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig = plt.figure(figsize=(22, 14.6), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 2200), ylim=(1460, 0))
    ax.axis("off")

    def label(x, y, value, size=13, color=TEXT, weight="normal", ha="left"):
        return ax.text(x, y, value, fontsize=size, color=color, weight=weight,
                       ha=ha, va="top", linespacing=1.5, zorder=5)

    def box(x, y, w, h, accent=BORDER, fill=PANEL, radius=16):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle=f"round,pad=0,rounding_size={radius}",
                     facecolor=fill, edgecolor=accent, linewidth=1.3, zorder=2))

    def line(points, color=BLUE, dashed=False, width=1.7):
        path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points)-1))
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=15,
                     linewidth=width, color=color, linestyle=(0, (5, 4)) if dashed else "-", zorder=3))

    def badge(x, y, text, color):
        label(x, y, text, size=10, color=color, weight="bold")

    label(60, 46, "SMART CASINO FLOOR / 2.0", 12, TEAL, "bold")
    label(60, 88, "Observe. Compare. Approve. Measure.", 30, TEXT, "bold")
    label(60, 153, "Implemented demo architecture · Streaming observations and one-click approved actions", 15, MUTED)

    # Main operational path. All observed state reaches the service through RW.
    badge(60, 235, "01  PHYSICAL STATE", TEAL)
    box(60, 272, 325, 295, TEAL)
    label(82, 295, "Floor simulator", 19, TEXT, "bold")
    label(82, 340, "36 positions · seats & queues\nTables, minimums & dealers\n10× clock · physical receipts", 13, MUTED)
    ax.plot([82, 363], [450, 450], color=BORDER, lw=1)
    label(82, 466, "Durable floor state", 12, TEAL, "bold")
    label(82, 499, "simulator-state / floor.json", 11, MUTED)

    badge(465, 235, "02  EVENT TRANSPORT", BLUE)
    box(465, 272, 290, 295)
    label(487, 295, "Apache Kafka", 19, TEXT, "bold")
    label(487, 344, "operational_events\ngaming_events\nfnb_events\nhotel_events", 13, MUTED)
    label(487, 503, "4 active input streams", 12, BLUE, "bold")

    badge(835, 235, "03  STREAMING SQL", BLUE)
    box(835, 272, 400, 760, BLUE, "#112439")
    label(859, 295, "RisingWave", 21, TEXT, "bold")
    box(859, 351, 352, 190, BORDER, "#172f49", 10)
    label(880, 372, "Operational state MVs", 15, BLUE, "bold")
    label(880, 414, "mv_ops_latest_snapshot\nmv_ops_table_state\nmv_ops_pit_state", 12, TEXT)
    label(859, 570, "Player feature MVs", 16, TEXT, "bold")
    label(859, 615, "5-minute windows · cumulative Theo\nLatest feature row per player", 12, MUTED)
    ax.plot([859, 1211], [705, 705], color=BORDER, lw=1)
    label(859, 736, "Prediction + rule views", 16, TEXT, "bold")
    label(859, 780, "recommendations_tbl\nActionable offers · VIP radar\nTheo by tier · historical analytics", 12, MUTED)
    ax.plot([859, 1211], [897, 897], color=BORDER, lw=1)
    label(859, 920, "risingwave-state", 12, BLUE, "bold")
    label(859, 956, "Catalog + MVs + chat_messages", 12, MUTED)

    badge(1315, 235, "04  DECISIONS & EXECUTION", TEAL)
    box(1315, 272, 380, 295, TEAL)
    label(1337, 295, "Operations service", 19, TEXT, "bold")
    label(1337, 340, "Constrained scenario engine\nPlan checks & automatic replanning\nOne-click approval & observations", 12, MUTED)
    ax.plot([1337, 1673], [450, 450], color=BORDER, lw=1)
    label(1337, 466, "SQLite WAL · operations-state", 12, TEAL, "bold")
    label(1337, 501, "Plans, tasks, commands & evidence", 11, MUTED)

    badge(1775, 235, "05  MANAGER WORKSPACE", TEAL)
    box(1775, 272, 365, 295, TEAL)
    label(1797, 295, "Streamlit · :8501", 19, TEXT, "bold")
    label(1797, 340, "Live / +15 / +30 minute floor\nGoal + constraint review\nAction center · evidence\nEnglish guided product tour", 12, MUTED)
    label(1797, 508, "Independent 3-second refresh", 11, TEAL, "bold")

    for start, end in [(385, 465), (755, 835), (1235, 1315), (1695, 1775)]:
        line([(start, 399), (end, 399)])
    label(425, 422, "events", 10, BLUE, ha="center")
    label(795, 422, "ingest", 10, BLUE, ha="center")
    label(1275, 422, "1s poll", 10, BLUE, ha="center")
    label(1735, 422, "HTTP", 10, BLUE, ha="center")

    # UI request path (no automatic approval).
    line([(2140, 320), (2170, 320), (2170, 208), (1725, 208), (1725, 320), (1695, 320)], GOLD, True)
    label(1915, 176, "Suggestions · one-click approval", 12, GOLD, ha="center")

    # Simulator pulls reviewed commands from the API. Bridge around RW to avoid
    # implying that this return route passes through streaming SQL or Kafka.
    command_route = [(1370, 567), (1370, 659), (1254, 659),
                     (1254, 1090), (355, 1090), (355, 567)]
    line(command_route, GOLD, True)
    label(490, 1057, "Reviewed command delivery · simulator polls /commands every ~1s", 12, GOLD)
    label(83, 601, "Validate → assign → apply", 11, TEAL)
    label(83, 635, "Confirmation returns in the\nnext streamed snapshot.", 11, MUTED)

    # Player ML is a separate analytics loop; it does not control operational tasks.
    badge(465, 724, "RETAINED PLAYER ML", PURPLE)
    box(465, 763, 290, 269, PURPLE)
    label(487, 788, "ML inference", 18, TEXT, "bold")
    label(487, 832, "Next game · churn\nOffer · VIP trajectory\nSynthetic scikit-learn models", 12, MUTED)
    label(487, 960, "Every 10s · SQL writeback", 11, PURPLE, "bold")
    line([(835, 681), (795, 681), (795, 817), (755, 817)], PURPLE)
    label(795, 711, "features", 10, PURPLE, ha="center")
    line([(755, 923), (788, 923), (788, 860), (835, 860)], PURPLE)
    label(788, 944, "INSERT", 10, PURPLE, ha="center")

    # The same model-provider category has two distinct callers and permissions.
    badge(1315, 724, "OPTIONAL MODEL PROVIDERS", PURPLE)
    box(1315, 763, 380, 269, PURPLE)
    label(1337, 788, "External LLM", 18, TEXT, "bold")
    label(1337, 833, "OpenAI · Claude · OpenRouter\nOps: goal → typed constraints\nChat: question → read-only SQL\nNo forecast math / action execution", 12, MUTED)
    label(1337, 958, "Templates + manual form without a key", 10.5, PURPLE)
    line([(1600, 567), (1600, 763)], PURPLE, True)
    line([(1625, 763), (1625, 567)], PURPLE, True)
    label(1565, 595, "Goal + constraints", 10, PURPLE, ha="right")
    label(1565, 628, "Validated JSON response", 10, PURPLE, ha="right")

    badge(1775, 724, "PLAYER ANALYTICS PAGE", BLUE)
    box(1775, 763, 365, 269, BLUE)
    label(1797, 788, "Analytics & SQL chat", 18, TEXT, "bold")
    label(1797, 833, "VIP radar · Theo · stored offers\nDirect RisingWave queries\nSeparate read-only SQL agent\nSQL-backed chat memory", 12, MUTED)
    label(1797, 986, "Historical players ≠ live seated count", 10.5, BLUE)
    line([(1958, 567), (1958, 763)], MUTED, True)
    label(1981, 651, "Page navigation", 10, MUTED)
    line([(1775, 898), (1695, 898)], PURPLE, True)
    line([(1695, 925), (1775, 925)], PURPLE, True)
    label(1735, 950, "SQL chat", 10, PURPLE, ha="center")

    # Direct SQL connection from the analytics page to RW, routed below the other
    # components. The generated SQL agent's SELECT boundary is distinct from the
    # application's writes for chat memory.
    line([(2030, 1032), (2030, 1150), (1145, 1150), (1145, 1032)], BLUE)
    label(1545, 1117, "Analytics SELECT queries · application-managed chat memory", 12, BLUE, ha="center")

    box(60, 1215, 2080, 174, BORDER, "#101d2e")
    label(86, 1239, "EXECUTION CONTRACT", 11, TEAL, "bold")
    label(86, 1273, "PENDING  →  EXECUTING  →  OBSERVING  →  CLOSED", 17, TEXT, "bold")
    label(86, 1320, "Freshness + version + staff + conflict checks  ·  Unique command IDs  ·  +5 / +15 minute observations", 12, MUTED)
    label(86, 1353, "One action per scenario. Forecasts use demo assumptions; observed changes are not causal revenue lift.", 11, MUTED)

    # Small, consistent legend; use labels as well as colors.
    line([(65, 1425), (112, 1425)], BLUE)
    label(125, 1415, "Stream / SQL / state", 10, MUTED)
    line([(435, 1425), (482, 1425)], GOLD, True)
    label(495, 1415, "Reviewed requests / commands", 10, MUTED)
    line([(865, 1425), (912, 1425)], PURPLE, True)
    label(925, 1415, "Optional LLM request / response", 10, MUTED)
    label(2140, 1415, "Local simulation · implemented 2.0", 10, MUTED, ha="right")

    fig.savefig(ROOT / "architecture.png", dpi=150, facecolor=BG)
    fig.savefig(ROOT / "architecture.svg", facecolor=BG,
                metadata={"Title": "Smart Casino Floor 2.0 architecture",
                          "Description": "Operational observations through Kafka and RisingWave, manager-reviewed command delivery, separate player analytics, and optional LLM providers."})
    svg = ROOT / "architecture.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    print("Generated architecture.png and architecture.svg")


if __name__ == "__main__":
    render()
