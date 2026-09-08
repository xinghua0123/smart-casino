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
CHAT_STORE_RETRY_INTERVAL = 15
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
}

# House edge constants shown in the explanation card (must match 02_feature_mvs.sql)
HOUSE_EDGES = {
    "Slots":     0.0750,
    "Baccarat":  0.0115,
    "Blackjack": 0.0075,
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


def _drop_chat_store_connection() -> None:
    conn = st.session_state.pop("chat_store_conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


def _mark_chat_store_unavailable() -> None:
    st.session_state.chat_store_unavailable = True
    st.session_state.chat_store_retry_at = time.monotonic() + CHAT_STORE_RETRY_INTERVAL
    _drop_chat_store_connection()


def _mark_chat_store_available() -> None:
    st.session_state.chat_store_unavailable = False
    st.session_state.pop("chat_store_retry_at", None)


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
    try:
        conn = get_chat_store_connection()
        _flush_pending_chat_deletions(conn)
        messages = CHAT_STORE.load_messages(conn, get_chat_session_id())
        st.session_state.chat_history_loaded = True
        _mark_chat_store_available()
        return messages
    except Exception:
        st.session_state.chat_history_loaded = False
        _mark_chat_store_unavailable()
        return []


def _flush_pending_chat_deletions(conn) -> None:
    pending = st.session_state.get("pending_chat_deletions", [])
    for session_id in pending:
        CHAT_STORE.clear_session(conn, session_id)
    st.session_state.pending_chat_deletions = []


def persist_chat_history() -> None:
    """Best-effort sync; session_state remains authoritative during outages."""
    try:
        conn = get_chat_store_connection()
        _flush_pending_chat_deletions(conn)
        for index, message in enumerate(st.session_state.chat_history):
            CHAT_STORE.append_message(
                conn, get_chat_session_id(), index, message
            )
        st.session_state.chat_history_loaded = True
        _mark_chat_store_available()
    except Exception:
        _mark_chat_store_unavailable()


def append_chat_message(message: dict) -> None:
    st.session_state.chat_history.append(message)
    persist_chat_history()


def clear_chat_session() -> None:
    old_session_id = get_chat_session_id()
    try:
        conn = get_chat_store_connection()
        CHAT_STORE.clear_session(conn, old_session_id)
        _flush_pending_chat_deletions(conn)
        _mark_chat_store_available()
    except Exception:
        pending = st.session_state.setdefault("pending_chat_deletions", [])
        if old_session_id not in pending:
            pending.append(old_session_id)
        _mark_chat_store_unavailable()

    st.session_state.chat_history = []
    st.session_state.chat_history_loaded = False
    st.session_state.chat_active = False
    st.session_state.pop("pending_question", None)
    st.session_state.chat_session_id = _new_chat_session_id()
    _set_query_param(CHAT_SESSION_PARAM, st.session_state.chat_session_id)


def retry_chat_store_if_due() -> None:
    if not st.session_state.get("chat_store_unavailable", False):
        return
    if time.monotonic() < st.session_state.get("chat_store_retry_at", 0):
        return

    if st.session_state.chat_history:
        persist_chat_history()
    else:
        st.session_state.chat_history = load_chat_history()


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
    st.session_state.chat_history = []
    st.session_state.chat_history_loaded = False
if not st.session_state.chat_history_loaded and not st.session_state.get(
    "chat_store_unavailable", False
):
    st.session_state.chat_history = load_chat_history()
retry_chat_store_if_due()
if "chat_active" not in st.session_state:
    st.session_state.chat_active = False

# ================================================================
# Sidebar: AI Chat Agent
# ================================================================
with st.sidebar:
    st.page_link("app.py", label="Back to floor operations", icon="🗺️")
    st.header("AI Data Agent")
    st.caption("Ask questions about the live casino data in natural language.")
    if st.session_state.get("chat_store_unavailable", False):
        st.warning(
            "Chat persistence is temporarily unavailable. This browser session "
            "will keep working and retry automatically."
        )

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
        st.caption("Analytics refresh independently every five seconds.")
        if st.button("Resume Auto-Refresh", use_container_width=True):
            st.session_state.chat_active = False
            st.rerun()


# ================================================================
# Main Dashboard
# ================================================================
@st.fragment(run_every="5s")
def analytics_live():
    st.title("Player Analytics")
    st.caption("Retained player history, synthetic ML recommendations and Theo analytics. Demo resets preserve these records; use Floor operations for current seats and queues.")

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
        FROM mv_player_latest_features
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

        kpis = [
            ("Recorded players", f"{int(cur['active_players']):,}", None, 0),
            ("Avg Bet", f"${cur_bet:,.0f}", f"{'-' if d_bet < 0 else '+'}${abs(d_bet):,.0f}" if abs(d_bet)>=1 else None, d_bet),
            ("Wagered (latest snapshots)", f"${cur_wagered:,.0f}", f"{'-' if d_wagered < 0 else '+'}${abs(d_wagered):,.0f}" if abs(d_wagered)>=1 else None, d_wagered),
            ("Theo (latest snapshots)", f"${cur_theo:,.0f}", f"{'-' if d_theo < 0 else '+'}${abs(d_theo):,.0f}" if abs(d_theo)>=1 else None, d_theo),
            ("Effective House Edge", f"{cur_edge:.2%}", f"{d_edge:+.2%}" if abs(d_edge)>=.0001 else None, d_edge),
        ]
        cards=[]
        for label,value,delta,change in kpis:
            help_text={
                "Theo (latest snapshots)": "Theoretical Win = sum of bet × house edge. Modeled expected gaming win, not actual profit.",
                "Effective House Edge": "Blended house edge for the actual game mix being played.",
            }.get(label, label)
            change_html=f'<div class="analytics-kpi-delta {"positive" if change>=0 else "negative"}">{"↑" if change>=0 else "↓"} {delta}</div>' if delta else ''
            cards.append(f'<div class="analytics-kpi" title="{help_text}"><div class="analytics-kpi-label">{label}</div><div class="analytics-kpi-value">{value}</div>{change_html}</div>')
        st.markdown("""<style>
        .analytics-kpis {display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,230px),1fr));gap:24px 20px;margin:20px 0;}
        .analytics-kpi {min-width:0;}
        .analytics-kpi-label {font-size:14px;margin-bottom:8px;}
        .analytics-kpi-value {font-size:32px;line-height:1.3;font-variant-numeric:tabular-nums;overflow-wrap:anywhere;}
        .analytics-kpi-delta {font-size:16px;margin-top:6px;overflow-wrap:anywhere;}
        .analytics-kpi-delta.positive {color:#09ab3b;}
        .analytics-kpi-delta.negative {color:#ff2b2b;}
        </style><div class="analytics-kpis">"""+''.join(cards)+'</div>',unsafe_allow_html=True)

        st.caption(
            f"Latest recorded window: {cur['latest_window']}  |  "
            f"Illustrative demo baseline: Avg Bet HKD {BASELINE['avg_bet']}, "
            f"Wagered/window HKD {BASELINE['wagered_per_5min']:,}, "
            f"Theo/window HKD {BASELINE['theo_per_5min']:,}, "
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
        st.subheader("Stored Recommendations")
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
            "Effective house edge = Theo Win ÷ Total Wagered — a player shifting from slots to baccarat "
            "or blackjack lowers this number even if they bet more."
        )
        st.caption(
            "Reinvestment tiers (offer_value): **40%** of theo for urgent retention, **35%** for VIP upgrade, "
            "**25%** for standard retention, **15%** for baseline loyalty."
        )

    st.divider()

    # ================================================================
    # Casino Floor Plan — live occupancy + starting-minimum recommendations
    # ================================================================
    st.page_link("app.py", label="Open live floor, operational actions and scenario planning", icon="🗺️")
    st.divider()

    st.subheader("Player Activity Distribution")
    bottom_left, bottom_right = st.columns(2)

    with bottom_left:
        game_dist = query("""
            SELECT 'Slots' AS game, AVG(pct_slots) AS pct FROM mv_player_latest_features
            UNION ALL SELECT 'Baccarat', AVG(pct_baccarat) FROM mv_player_latest_features
            UNION ALL SELECT 'Blackjack', AVG(pct_blackjack) FROM mv_player_latest_features
        """)
        if not game_dist.empty:
            fig3 = px.pie(game_dist, values="pct", names="game", title="Game Type Distribution")
            fig3.update_layout(height=300, margin=dict(t=40, b=20))
            st.plotly_chart(fig3, use_container_width=True)

    with bottom_right:
        tier_dist = query("""
            SELECT tier, COUNT(DISTINCT player_id) AS players
            FROM mv_player_latest_features GROUP BY tier ORDER BY players DESC
        """)
        if not tier_dist.empty:
            fig4 = px.bar(tier_dist, x="tier", y="players", title="Players by Tier", color="tier")
            fig4.update_layout(height=300, margin=dict(t=40, b=20), showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)

    # ================================================================
    # Post-render: process pending chat question (after main dashboard renders)
    # ================================================================

analytics_live()

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
