"""
Smart Casino Floor Dashboard — real-time view of ML-powered recommendations,
high-roller radar, churn alerts, and player activity.

Queries RisingWave materialized views and auto-refreshes every 5 seconds.
Includes an AI chat agent in the sidebar for natural language data exploration.
"""

import os
import time
import uuid

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import streamlit as st

from agent import create_agent
from chat_store import SqlChatStore

RW_HOST = os.environ.get("RISINGWAVE_HOST", "localhost")
RW_PORT = os.environ.get("RISINGWAVE_PORT", "4566")
RW_USER = os.environ.get("RISINGWAVE_USER", "root")
RW_DBNAME = os.environ.get("RISINGWAVE_DBNAME", "dev")
REFRESH_INTERVAL = 5
CHAT_SESSION_PARAM = "chat_session"
CHAT_STORE = SqlChatStore.from_env(
    default_host=RW_HOST,
    default_port=RW_PORT,
    default_user=RW_USER,
    default_dbname=RW_DBNAME,
)

# --- Last month's daily average baseline (hardcoded) ---
BASELINE = {
    "avg_bet": 185,
    "spend_per_min": 210,
    "wagered_per_5min": 150_000,
    "theo_per_5min": 7_500,        # ~5% blended house edge × 150K wagered
    "house_edge": 0.0500,          # 5.00% blended across the typical game mix
    "win_rate": 0.42,
    "label": "vs last month avg",
}

# House edge constants shown in the explanation card (must match 02_feature_mvs.sql)
HOUSE_EDGES = {
    "Slots":     0.0750,
    "Roulette":  0.0526,
    "Blackjack": 0.0075,
    "Poker":     0.0250,
}

EXAMPLE_QUESTIONS = [
    "Who are the top 5 high-roller candidates?",
    "What's the average churn risk by tier?",
    "Which game type has the highest average bet?",
    "How many players need urgent retention?",
    "Show me emerging archetype players with HR score above 0.6",
    "What offers are recommended for gold tier players?",
]


def get_connection():
    if "rw_conn" not in st.session_state or st.session_state.rw_conn.closed:
        st.session_state.rw_conn = psycopg2.connect(
            host=RW_HOST, port=RW_PORT, user=RW_USER, dbname=RW_DBNAME
        )
        st.session_state.rw_conn.autocommit = True
    return st.session_state.rw_conn


def get_chat_store_connection():
    if "chat_store_conn" not in st.session_state or st.session_state.chat_store_conn.closed:
        st.session_state.chat_store_conn = CHAT_STORE.connect()
        CHAT_STORE.ensure_schema(st.session_state.chat_store_conn)
    return st.session_state.chat_store_conn


def _get_query_param(name: str) -> str | None:
    value = st.query_params.get(name)
    if isinstance(value, list):
        return value[0] if value else None
    return value or None


def _set_query_param(name: str, value: str) -> None:
    st.query_params[name] = value


def _new_chat_session_id() -> str:
    return uuid.uuid4().hex


def get_chat_session_id() -> str:
    if "chat_session_id" not in st.session_state:
        session_id = _get_query_param(CHAT_SESSION_PARAM) or _new_chat_session_id()
        st.session_state.chat_session_id = session_id
        _set_query_param(CHAT_SESSION_PARAM, session_id)
    return st.session_state.chat_session_id


def load_chat_history() -> list[dict]:
    conn = get_chat_store_connection()
    return CHAT_STORE.load_messages(conn, get_chat_session_id())


def append_chat_message(message: dict) -> None:
    st.session_state.chat_history.append(message)
    conn = get_chat_store_connection()
    CHAT_STORE.append_message(
        conn,
        get_chat_session_id(),
        len(st.session_state.chat_history) - 1,
        message,
    )


def clear_chat_session() -> None:
    conn = get_chat_store_connection()
    CHAT_STORE.clear_session(conn, get_chat_session_id())
    st.session_state.chat_history = []
    st.session_state.chat_active = False
    st.session_state.pop("pending_question", None)
    st.session_state.chat_session_id = _new_chat_session_id()
    _set_query_param(CHAT_SESSION_PARAM, st.session_state.chat_session_id)


def query(sql: str) -> pd.DataFrame:
    try:
        conn = get_connection()
        return pd.read_sql(sql, conn)
    except Exception as e:
        st.error(f"Query error: {e}")
        if "rw_conn" in st.session_state:
            try:
                st.session_state.rw_conn.close()
            except Exception:
                pass
            del st.session_state.rw_conn
        return pd.DataFrame()


# --- Page config ---
st.set_page_config(page_title="Smart Casino Floor", layout="wide")

_ = get_chat_session_id()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()
if "chat_active" not in st.session_state:
    st.session_state.chat_active = False

# ================================================================
# Sidebar: AI Chat Agent
# ================================================================
with st.sidebar:
    st.header("AI Data Agent")
    st.caption("Ask questions about the live casino data in natural language.")

    # --- LLM provider config ---
    provider = st.selectbox("LLM Provider", ["Claude", "OpenAI", "OpenRouter", "Azure OpenAI"],
                            key="llm_provider")
    api_key = os.environ.get("LLM_API_KEY", "")
    api_key = st.text_input("API Key", value=api_key, type="password", key="llm_api_key")

    if provider == "Claude":
        claude_base_url = st.text_input(
            "Base URL (optional)", key="claude_base_url",
            placeholder="https://api.anthropic.com  (or a proxy)",
            help="Leave blank for api.anthropic.com. Set a custom URL to use an Anthropic-compatible proxy (e.g. PackyAPI).",
        )
        claude_model = st.text_input("Model", value="claude-sonnet-4-20250514", key="claude_model")
    elif provider == "OpenAI":
        openai_base_url = st.text_input(
            "Base URL (optional)", key="openai_base_url",
            placeholder="https://api.openai.com/v1  (or a proxy like PackyAPI)",
            help="Leave blank for api.openai.com. Set a custom URL to use an OpenAI-compatible proxy (e.g. PackyAPI, OpenRouter, LiteLLM).",
        )
        openai_model = st.text_input("Model", value="gpt-4o", key="openai_model")
    elif provider == "OpenRouter":
        st.caption("Routes to `https://openrouter.ai/api/v1`. Use OpenRouter model slugs like `openai/gpt-4o-mini`, `anthropic/claude-sonnet-4`, `google/gemini-2.0-flash-exp`.")
        openrouter_model = st.text_input(
            "Model", value="openai/gpt-4o-mini", key="openrouter_model",
            help="See https://openrouter.ai/models for the full catalog.",
        )
    elif provider == "Azure OpenAI":
        azure_url = st.text_input("Azure Endpoint URL", key="azure_url",
                                  placeholder="https://your-resource.openai.azure.com/")
        azure_model = st.text_input("Deployment Name", value="gpt-4o", key="azure_model")

    st.divider()

    # --- Display chat history ---
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sql"):
                with st.expander("SQL Query"):
                    st.code(msg["sql"], language="sql")
            if msg.get("data") is not None and not msg["data"].empty:
                with st.expander("Raw Data"):
                    st.dataframe(msg["data"], hide_index=True, use_container_width=True)

    # --- Example questions ---
    if not st.session_state.chat_history:
        st.markdown("**Try asking:**")
        for eq in EXAMPLE_QUESTIONS:
            if st.button(eq, key=f"eq_{eq}", use_container_width=True):
                st.session_state.pending_question = eq
                st.rerun()

    # --- Chat input ---
    user_input = st.chat_input("Ask about the casino data...", key="chat_input")

    # Check for pending question from example buttons
    if "pending_question" not in st.session_state and user_input:
        st.session_state.pending_question = user_input

    # If a question is pending, show user message + thinking placeholder immediately
    if "pending_question" in st.session_state:
        q = st.session_state.pending_question
        # Append user message exactly once
        if not (st.session_state.chat_history and
                st.session_state.chat_history[-1].get("role") == "user" and
                st.session_state.chat_history[-1].get("content") == q):
            append_chat_message({"role": "user", "content": q})
            st.session_state.chat_active = True
            with st.chat_message("user"):
                st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown("_Thinking…_")

    # --- Clear chat button ---
    if st.session_state.chat_history:
        if st.button("Clear Chat", use_container_width=True):
            clear_chat_session()
            st.rerun()

    # --- Resume auto-refresh toggle ---
    if st.session_state.chat_active:
        st.caption("Dashboard auto-refresh is paused while chatting.")
        if st.button("Resume Auto-Refresh", use_container_width=True):
            st.session_state.chat_active = False
            st.rerun()


# ================================================================
# Main Dashboard
# ================================================================
st.title("Smart Casino Floor")
st.caption("Real-time ML-powered gaming recommendations & high-roller detection")

# --- Top-level KPIs ---
cur_df = query("""
    SELECT
        COUNT(DISTINCT player_id) AS active_players,
        AVG(avg_bet) AS avg_bet,
        SUM(total_bet) AS total_wagered,
        SUM(theo_win_window) AS theo_win_window,
        AVG(effective_house_edge) AS avg_house_edge,
        AVG(win_rate) AS avg_win_rate,
        MAX(window_end) AS latest_window
    FROM mv_player_features
    WHERE window_start = (SELECT MAX(window_start) FROM mv_player_features)
""")

if not cur_df.empty and cur_df["active_players"].iloc[0] is not None and int(cur_df["active_players"].iloc[0]) > 0:
    cur = cur_df.iloc[0]
    cur_bet = float(cur["avg_bet"])
    cur_wagered = float(cur["total_wagered"])
    cur_theo = float(cur["theo_win_window"] or 0.0)
    cur_edge = float(cur["avg_house_edge"] or 0.0)
    cur_wr = float(cur["avg_win_rate"])

    d_bet = cur_bet - BASELINE["avg_bet"]
    d_wagered = cur_wagered - BASELINE["wagered_per_5min"]
    d_theo = cur_theo - BASELINE["theo_per_5min"]
    d_edge = cur_edge - BASELINE["house_edge"]
    lbl = BASELINE["label"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Active Players", int(cur["active_players"]))
    c2.metric("Avg Bet", f"${cur_bet:,.0f}",
              delta=f"${d_bet:+,.0f} {lbl}" if abs(d_bet) >= 1 else None)
    c3.metric("Total Wagered (window)", f"${cur_wagered:,.0f}",
              delta=f"${d_wagered:+,.0f} {lbl}" if abs(d_wagered) >= 1 else None)
    c4.metric("Theo Win (window)", f"${cur_theo:,.0f}",
              delta=f"${d_theo:+,.0f} {lbl}" if abs(d_theo) >= 1 else None,
              help="Theoretical Win = Σ(bet × house_edge). Casino's expected profit from this window's play, independent of short-term luck.")
    c5.metric("Effective House Edge", f"{cur_edge:.2%}",
              delta=f"{d_edge:+.2%} {lbl}" if abs(d_edge) >= 0.0001 else None,
              delta_color="normal",
              help="Blended house edge given the actual game mix being played. Higher = more profitable game mix.")

    st.caption(
        f"Current window: {cur['latest_window']}  |  "
        f"Baseline (last month daily avg): Avg Bet ${BASELINE['avg_bet']}, "
        f"Wagered/window ${BASELINE['wagered_per_5min']:,}, "
        f"Theo/window ${BASELINE['theo_per_5min']:,}, "
        f"House Edge {BASELINE['house_edge']:.2%}"
    )
else:
    st.info("Waiting for player data to arrive...")

st.divider()

# --- Charts ---
left, right = st.columns(2)

with left:
    st.subheader("High Roller Radar")
    radar_df = query("""
        SELECT player_id, tier, archetype,
               ROUND(high_roller_similarity::numeric, 3) AS similarity,
               ROUND(avg_bet::numeric, 0) AS avg_bet,
               ROUND(cumulative_gaming_spend::numeric, 0) AS total_spend,
               ROUND(cumulative_theo_win::numeric, 0) AS theo_win,
               ROUND(spend_per_minute::numeric, 1) AS spend_per_min,
               category_diversity
        FROM mv_high_roller_radar
        ORDER BY high_roller_similarity DESC
        LIMIT 15
    """)
    if not radar_df.empty:
        fig = px.scatter(
            radar_df, x="spend_per_min", y="similarity", size="avg_bet",
            color="tier", hover_data=["player_id", "archetype", "total_spend"],
            title="Similarity vs. Spend Velocity",
            labels={"spend_per_min": "Spend per Minute ($)", "similarity": "High Roller Similarity"},
        )
        fig.update_layout(height=350, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No emerging high-roller candidates detected yet...")

with right:
    st.subheader("Active Recommendations")
    action_summary = query("""
        SELECT action_type, COUNT(*) AS cnt
        FROM mv_actionable_recommendations GROUP BY action_type ORDER BY cnt DESC
    """)
    if not action_summary.empty:
        fig2 = px.bar(
            action_summary, x="action_type", y="cnt", color="action_type",
            title="Recommendation Actions (All Players)",
            labels={"action_type": "Action Type", "cnt": "Players"},
            color_discrete_map={
                "URGENT_RETENTION": "#e74c3c", "VIP_UPGRADE_CANDIDATE": "#f39c12",
                "RETENTION_OFFER": "#3498db", "STANDARD_RECOMMENDATION": "#2ecc71",
            },
        )
        fig2.update_layout(height=350, margin=dict(t=40, b=20), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

# --- Explanations ---
left, right = st.columns(2)
with left:
    st.markdown("""
**How to read the Radar:**
- **HR Score** = behavioral similarity to known high rollers (0-1). Factors: bet size, table game preference, cross-category spend, spend velocity.
- **Higher & righter** on the chart = stronger high-roller signal. These are players who *aren't* VIPs yet but *behave* like ones.
- **Action:** HR Score > 0.7 = proactive VIP outreach. Score 0.4-0.7 = monitor closely.
    """)
with right:
    st.markdown("""
**Recommendation categories:**

Offer value = reinvestment % of the player's **cumulative Theo Win** (industry-standard comp sizing), with an `avg_bet` floor so new players still get a sensible offer.

- **URGENT_RETENTION** — Churn risk > 45% AND Silver+ tier. Action: host call, aggressive save offer — **40% of cumulative theo**.
- **VIP_UPGRADE_CANDIDATE** — ML predicts high-roller trajectory. Action: VIP invite, comp upgrade — **35% of cumulative theo**.
- **RETENTION_OFFER** — Churn risk > 38%. Action: targeted offer matching preferred reward type — **25% of cumulative theo**.
- **STANDARD_RECOMMENDATION** — Healthy player. Action: cross-sell next-best-game — **15% of cumulative theo** (baseline loyalty).
    """)

# --- Data tables ---
left, right = st.columns(2)
with left:
    if not radar_df.empty:
        def hr_color(val):
            if val >= 0.7:
                return "background-color: rgba(243, 156, 18, 0.3)"
            elif val >= 0.55:
                return "background-color: rgba(243, 156, 18, 0.15)"
            return ""
        styled_radar = radar_df.rename(columns={
            "player_id": "Player", "tier": "Tier", "similarity": "HR Score",
            "avg_bet": "Avg Bet", "total_spend": "Total Spend",
            "theo_win": "Theo Win", "spend_per_min": "$/min",
            "category_diversity": "Categories", "archetype": "Type",
        })
        st.dataframe(
            styled_radar.style.applymap(hr_color, subset=["HR Score"]),
            hide_index=True, use_container_width=True,
        )

with right:
    recs_df = query("""
        SELECT player_id, next_best_game, action_type, offer_sensitivity,
               ROUND(churn_probability::numeric, 3) AS churn_prob,
               ROUND(high_roller_score::numeric, 3) AS hr_score,
               high_roller_trajectory AS hr_trajectory, tier,
               ROUND(cumulative_theo_win::numeric, 0) AS theo_win,
               ROUND(offer_value::numeric, 0) AS offer_value
        FROM mv_actionable_recommendations
        ORDER BY
            CASE action_type WHEN 'URGENT_RETENTION' THEN 1
                WHEN 'VIP_UPGRADE_CANDIDATE' THEN 2
                WHEN 'RETENTION_OFFER' THEN 3 ELSE 4 END,
            churn_probability DESC
        LIMIT 20
    """)
    if not recs_df.empty:
        st.dataframe(
            recs_df.rename(columns={
                "player_id": "Player", "next_best_game": "Suggested Game",
                "action_type": "Action", "churn_prob": "Churn Risk",
                "hr_score": "HR Score", "hr_trajectory": "HR Track",
                "tier": "Tier", "theo_win": "Theo Win",
                "offer_value": "Offer $", "offer_sensitivity": "Best Offer",
            }),
            hide_index=True, use_container_width=True,
        )
    else:
        st.info("Waiting for ML predictions...")

st.divider()

# --- Theo Win section ---
st.subheader("Theoretical Win & House Advantage")
theo_left, theo_mid, theo_right = st.columns([1.2, 1.2, 1])

with theo_left:
    theo_tier_df = query("""
        SELECT tier,
               players,
               ROUND(total_theo_win::numeric, 0)       AS total_theo,
               ROUND(avg_theo_per_player::numeric, 0)  AS avg_theo,
               ROUND(avg_effective_house_edge::numeric, 4) AS avg_edge
        FROM mv_theo_by_tier
        ORDER BY
            CASE tier WHEN 'diamond' THEN 1 WHEN 'platinum' THEN 2
                      WHEN 'gold' THEN 3 WHEN 'silver' THEN 4
                      WHEN 'bronze' THEN 5 ELSE 6 END
    """)
    if not theo_tier_df.empty:
        fig_theo = px.bar(
            theo_tier_df, x="tier", y="total_theo", color="tier",
            title="Cumulative Theo Win by Tier",
            labels={"tier": "Tier", "total_theo": "Total Theo Win ($)"},
            color_discrete_map={
                "diamond": "#b9f2ff", "platinum": "#e5e4e2", "gold": "#f1c40f",
                "silver": "#bdc3c7", "bronze": "#cd7f32",
            },
        )
        fig_theo.update_layout(height=320, margin=dict(t=40, b=20), showlegend=False)
        st.plotly_chart(fig_theo, use_container_width=True)

with theo_mid:
    if not theo_tier_df.empty:
        fig_edge = px.bar(
            theo_tier_df, x="tier", y="avg_edge", color="tier",
            title="Avg Effective House Edge by Tier",
            labels={"tier": "Tier", "avg_edge": "Effective House Edge"},
            color_discrete_map={
                "diamond": "#b9f2ff", "platinum": "#e5e4e2", "gold": "#f1c40f",
                "silver": "#bdc3c7", "bronze": "#cd7f32",
            },
        )
        fig_edge.update_yaxes(tickformat=".2%")
        fig_edge.update_layout(height=320, margin=dict(t=40, b=20), showlegend=False)
        st.plotly_chart(fig_edge, use_container_width=True)

with theo_right:
    st.markdown("**House Advantage reference**")
    edge_lines = "\n".join([f"- **{g}** — {e:.2%}" for g, e in HOUSE_EDGES.items()])
    st.markdown(edge_lines)
    st.caption(
        "**Theo Win** = Σ(bet × house_edge). The casino's expected profit regardless of short-term luck. "
        "Effective house edge = Theo Win ÷ Total Wagered — a player shifting from slots to blackjack "
        "lowers this number even if they bet more."
    )
    st.caption(
        "Reinvestment tiers (offer_value): **40%** of theo for urgent retention, **35%** for VIP upgrade, "
        "**25%** for standard retention, **15%** for baseline loyalty."
    )

st.divider()

# ================================================================
# Casino Floor Plan — per-table live view + raise/lower-limit recommendations
# ================================================================
st.subheader("Casino Floor Plan")
st.caption(
    "Live map of the gaming floor. Each marker is one table — shape = game type, "
    "color = recommended action (raise / lower limit, hot, cold, hold). "
    "Hover a table for live stats (active players, avg bet, theo win, suggested new limits). "
    "Rules use foot traffic + bet segmentation — tune the thresholds to match the property."
)

floor_df = query("""
    SELECT table_id, game_type, table_x, table_y,
           limit_min, limit_max,
           active_players, bets,
           ROUND(avg_bet::numeric, 0)         AS avg_bet,
           ROUND(max_bet::numeric, 0)         AS max_bet,
           ROUND(total_bet::numeric, 0)       AS total_bet,
           ROUND(theo_win_window::numeric, 0) AS theo_win_window,
           action_type,
           suggested_limit_min,
           suggested_limit_max
    FROM mv_table_recommendations
""")

ACTION_COLORS = {
    "RAISE_LIMIT":  "#e74c3c",  # red — push the ceiling up
    "LOWER_LIMIT":  "#3498db",  # blue — drop the floor to attract traffic
    "HOT":          "#f39c12",  # orange — healthy, keep an eye on it
    "COLD":         "#7f8c8d",  # grey — empty
    "HOLD":         "#2ecc71",  # green — balanced, no change needed
}
GAME_SYMBOLS = {
    "slots":     "square",
    "blackjack": "circle",
    "roulette":  "diamond",
    "poker":     "hexagon",
}

if not floor_df.empty:
    floor_left, floor_right = st.columns([2, 1])

    with floor_left:
        # Plotly scatter: tables positioned on the floor (x, y from tables_dim)
        plot_df = floor_df.copy()
        plot_df["limit_label"] = plot_df.apply(
            lambda r: f"${int(r['limit_min'])}-${int(r['limit_max'])}", axis=1
        )
        plot_df["suggested_label"] = plot_df.apply(
            lambda r: f"${int(r['suggested_limit_min'])}-${int(r['suggested_limit_max'])}",
            axis=1,
        )

        fig_floor = px.scatter(
            plot_df,
            x="table_x", y="table_y",
            color="action_type",
            symbol="game_type",
            text="table_id",
            color_discrete_map=ACTION_COLORS,
            symbol_map=GAME_SYMBOLS,
            category_orders={
                "action_type": ["RAISE_LIMIT", "LOWER_LIMIT", "HOT", "COLD", "HOLD"],
            },
            hover_data={
                "table_id": True,
                "game_type": True,
                "active_players": True,
                "bets": True,
                "avg_bet": ":$,.0f",
                "max_bet": ":$,.0f",
                "theo_win_window": ":$,.0f",
                "limit_label": True,
                "suggested_label": True,
                "action_type": True,
                "table_x": False,
                "table_y": False,
                "limit_min": False,
                "limit_max": False,
                "suggested_limit_min": False,
                "suggested_limit_max": False,
                "total_bet": False,
            },
            labels={
                "table_x": "",
                "table_y": "",
                "action_type": "Action",
                "game_type": "Game",
                "limit_label": "Current limit",
                "suggested_label": "Suggested limit",
                "avg_bet": "Avg bet",
                "max_bet": "Max bet",
                "theo_win_window": "Theo Win",
                "active_players": "Active players",
                "bets": "Bets",
            },
            title="Live Floor Map — tables by position, color = recommended action",
        )
        # Uniform marker size — small enough to never overlap given the 1.0+ unit
        # grid spacing in tables_dim.
        fig_floor.update_traces(
            marker=dict(size=18, line=dict(width=1, color="#111")),
            textposition="bottom center",
            textfont=dict(size=9, color="#bdc3c7"),
        )
        # Background zones for pit labels (positioned between clusters)
        fig_floor.add_annotation(x=2.4, y=9.7, text="SLOTS (penny)", showarrow=False,
                                 font=dict(size=11, color="#95a5a6"))
        fig_floor.add_annotation(x=2.4, y=2.7, text="SLOTS (standard)", showarrow=False,
                                 font=dict(size=11, color="#95a5a6"))
        fig_floor.add_annotation(x=6.4, y=5.9, text="BLACKJACK pit", showarrow=False,
                                 font=dict(size=11, color="#95a5a6"))
        fig_floor.add_annotation(x=6.4, y=8.5, text="High-limit BJ", showarrow=False,
                                 font=dict(size=11, color="#95a5a6"))
        fig_floor.add_annotation(x=7.6, y=0.0, text="ROULETTE", showarrow=False,
                                 font=dict(size=11, color="#95a5a6"))
        fig_floor.add_annotation(x=10.1, y=2.7, text="POKER lounge", showarrow=False,
                                 font=dict(size=11, color="#95a5a6"))
        fig_floor.update_xaxes(visible=False, range=[-0.2, 11.5])
        fig_floor.update_yaxes(visible=False, range=[-0.3, 10.0], scaleanchor="x", scaleratio=1)
        fig_floor.update_layout(
            height=520, margin=dict(t=40, b=10, l=10, r=10),
            plot_bgcolor="#0e1117",
            legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_floor, use_container_width=True)

    with floor_right:
        st.markdown("**Action meanings**")
        st.markdown(
            "- 🔴 **RAISE_LIMIT** — packed (≥ 3 players) and avg bet is ≥ 70% of the "
            "current ceiling. Push the max up to capture more theo.\n"
            "- 🔵 **LOWER_LIMIT** — cold (≤ 1 player, fewer than 8 bets) and the min "
            "is above entry-level. Drop it to pull in casual traffic.\n"
            "- 🟠 **HOT** — healthy occupancy generating strong theo this window.\n"
            "- ⚫ **COLD** — essentially no activity. Watch; no limit change yet.\n"
            "- 🟢 **HOLD** — balanced. Leave it alone."
        )
        st.caption(
            "Rules are a starting point. Casinos typically layer in weather, holidays, "
            "player VIP mix, and nearby-table overflow — tune the thresholds in "
            "`mv_table_recommendations` per property."
        )

        # Priority list of tables that need action
        action_priority = {
            "RAISE_LIMIT": 1, "LOWER_LIMIT": 2, "HOT": 3, "COLD": 4, "HOLD": 5,
        }
        priority_df = floor_df.copy()
        priority_df["priority"] = priority_df["action_type"].map(action_priority)
        priority_df = (
            priority_df[priority_df["action_type"].isin(["RAISE_LIMIT", "LOWER_LIMIT"])]
            .sort_values(["priority", "theo_win_window"], ascending=[True, False])
        )
        if not priority_df.empty:
            st.markdown("**Needs attention now**")
            show_cols = ["table_id", "game_type", "action_type", "active_players",
                         "avg_bet", "limit_min", "limit_max",
                         "suggested_limit_min", "suggested_limit_max"]
            st.dataframe(
                priority_df[show_cols].rename(columns={
                    "table_id": "Table", "game_type": "Game",
                    "action_type": "Action", "active_players": "Players",
                    "avg_bet": "Avg Bet", "limit_min": "Min",
                    "limit_max": "Max", "suggested_limit_min": "→ Min",
                    "suggested_limit_max": "→ Max",
                }),
                hide_index=True, use_container_width=True, height=260,
            )
        else:
            st.info("No tables currently flagged for a limit change — floor is balanced.")
else:
    st.info("Waiting for per-table data (first window finishes in ~5 min after restart)...")

st.divider()

# --- Bottom: distributions ---
st.subheader("Player Activity Distribution")
bottom_left, bottom_right = st.columns(2)

with bottom_left:
    game_dist = query("""
        SELECT 'Slots' AS game, AVG(pct_slots) AS pct FROM mv_player_features
        UNION ALL SELECT 'Roulette', AVG(pct_roulette) FROM mv_player_features
        UNION ALL SELECT 'Blackjack', AVG(pct_blackjack) FROM mv_player_features
        UNION ALL SELECT 'Poker', AVG(pct_poker) FROM mv_player_features
    """)
    if not game_dist.empty:
        fig3 = px.pie(game_dist, values="pct", names="game", title="Game Type Distribution")
        fig3.update_layout(height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig3, use_container_width=True)

with bottom_right:
    tier_dist = query("""
        SELECT tier, COUNT(DISTINCT player_id) AS players
        FROM mv_player_features GROUP BY tier ORDER BY players DESC
    """)
    if not tier_dist.empty:
        fig4 = px.bar(tier_dist, x="tier", y="players", title="Players by Tier", color="tier")
        fig4.update_layout(height=300, margin=dict(t=40, b=20), showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

# ================================================================
# Post-render: process pending chat question (after main dashboard renders)
# ================================================================
if "pending_question" in st.session_state:
    q = st.session_state.pop("pending_question")
    if not api_key:
        append_chat_message({
            "role": "assistant",
            "content": "Please enter your API key in the sidebar to use the chat agent.",
        })
    else:
        try:
            kwargs = {}
            if provider == "Claude":
                base = (st.session_state.get("claude_base_url") or "").strip()
                if base:
                    kwargs["base_url"] = base
                kwargs["model"] = st.session_state.get("claude_model", "claude-sonnet-4-20250514")
            elif provider == "OpenAI":
                base = (st.session_state.get("openai_base_url") or "").strip()
                if base:
                    kwargs["base_url"] = base
                kwargs["model"] = st.session_state.get("openai_model", "gpt-4o")
            elif provider == "OpenRouter":
                kwargs["model"] = st.session_state.get("openrouter_model", "openai/gpt-4o-mini")
            elif provider == "Azure OpenAI":
                kwargs["base_url"] = st.session_state.get("azure_url", "")
                kwargs["model"] = st.session_state.get("azure_model", "gpt-4o")

            agent = create_agent(provider, api_key, **kwargs)
            conn = get_connection()
            history_for_agent = st.session_state.chat_history[:-1]
            result = agent.ask(q, conn, history=history_for_agent)

            if result.get("error"):
                append_chat_message({
                    "role": "assistant",
                    "content": f"Error: {result['error']}",
                    "sql": result.get("sql"),
                })
            else:
                append_chat_message({
                    "role": "assistant",
                    "content": result["answer"],
                    "sql": result.get("sql"),
                    "data": result.get("data"),
                })
        except Exception as e:
            append_chat_message({
                "role": "assistant",
                "content": f"Agent error: {e}",
            })
    st.rerun()

# --- Auto-refresh (paused when chat is active) ---
if not st.session_state.get("chat_active", False):
    time.sleep(REFRESH_INTERVAL)
    st.rerun()
