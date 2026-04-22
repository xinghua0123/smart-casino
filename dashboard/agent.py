"""
Pluggable LLM chat agent for the Smart Casino Floor dashboard.

Translates natural language questions into SQL queries against RisingWave,
executes them, and returns human-readable answers.

Supports: Claude (default), OpenAI, Azure OpenAI.
"""

import json
import re
from abc import ABC, abstractmethod

import pandas as pd
import psycopg2

# --- Schema context embedded in every system prompt ---
SCHEMA_CONTEXT = """
You are a data analyst assistant for a real-time casino analytics dashboard powered by RisingWave (a streaming SQL database).

## Available Tables and Materialized Views

### mv_player_features
Per-player, per-5-min-window gaming features (the main feature store).
Columns: player_id (VARCHAR), window_start (TIMESTAMPTZ), window_end (TIMESTAMPTZ), games_played (INT), avg_bet (FLOAT), total_bet (FLOAT), total_payout (FLOAT), win_rate (FLOAT 0-1), pct_slots (FLOAT 0-1), pct_baccarat (FLOAT 0-1), pct_roulette (FLOAT 0-1), pct_blackjack (FLOAT 0-1), pct_poker (FLOAT 0-1), cumulative_gaming_spend (FLOAT), tier (VARCHAR: bronze/silver/gold/platinum/diamond), archetype (VARCHAR: casual/regular/high_roller/emerging), theo_win_window (FLOAT — theoretical win in this window), cumulative_theo_win (FLOAT — running total theo win for this player), effective_house_edge (FLOAT 0-1 — blended house edge given game mix), fnb_orders (INT), fnb_spend (FLOAT), cumulative_fnb_spend (FLOAT), hotel_events (INT), hotel_spend (FLOAT), hotel_action_types (INT), category_diversity (INT 1-3), spend_per_minute (FLOAT)

### mv_player_theo_cumulative
Running per-player lifetime totals (no window — updated continuously).
Columns: player_id, cumulative_theo_win (FLOAT), cumulative_wagered (FLOAT), effective_house_edge (FLOAT 0-1)

### mv_player_high_roller_similarity
Per-player, per-5-min-window high-roller similarity score (0-1). Higher = more similar to known VIPs. This view contains multiple rows per player over time.
Columns: player_id, window_start, tier, archetype, avg_bet, cumulative_gaming_spend, cumulative_fnb_spend, spend_per_minute, category_diversity, theo_win_window, cumulative_theo_win, effective_house_edge, high_roller_similarity (FLOAT 0-1)

### mv_high_roller_radar
Filtered current-candidate view: only non-VIP players with high_roller_similarity > 0.4, keeping only the latest row per player_id. Use this for unique emerging VIP candidates.
Columns: player_id, tier, archetype, high_roller_similarity, avg_bet, cumulative_gaming_spend, cumulative_fnb_spend, spend_per_minute, category_diversity, theo_win_window, cumulative_theo_win, effective_house_edge, window_start

### mv_actionable_recommendations
ML predictions joined with business rules. One row per player.
Columns: player_id, next_best_game (VARCHAR), churn_probability (FLOAT 0-1), offer_sensitivity (VARCHAR: free_play/fnb_voucher/hotel_upgrade/cashback), high_roller_score (FLOAT 0-1), high_roller_trajectory (BOOLEAN), tier, avg_bet, cumulative_gaming_spend, theo_win_window, cumulative_theo_win, effective_house_edge, action_type (VARCHAR: STANDARD_RECOMMENDATION/RETENTION_OFFER/VIP_UPGRADE_CANDIDATE/URGENT_RETENTION), offer_value (FLOAT — reinvestment $), recommendation_ts (TIMESTAMPTZ)

### recommendations_tbl
Raw ML predictions (updated every 10s by the inference service).
Columns: player_id (PK), next_best_game, churn_probability, offer_sensitivity, high_roller_score, high_roller_trajectory, ts

### mv_theo_by_tier
Aggregated theo-win metrics grouped by tier.
Columns: tier, players (INT), total_theo_win (FLOAT), avg_theo_per_player (FLOAT), avg_effective_house_edge (FLOAT 0-1)

### mv_dashboard_stats
Aggregated dashboard KPIs (single row).
Columns: active_players (INT), avg_bet_all (FLOAT), avg_spend_per_min (FLOAT), total_wagered (FLOAT), theo_win_window (FLOAT), avg_house_edge (FLOAT 0-1)

### tables_dim
Static dimension table: one row per physical table on the casino floor.
Columns: table_id (VARCHAR PK), game_type (VARCHAR: slots/baccarat/blackjack/roulette/poker), table_x (FLOAT — floor plan x), table_y (FLOAT — floor plan y), limit_min (FLOAT — current min bet), limit_max (FLOAT — current max bet)

### mv_table_activity
Per-table per-5-min-window rollup of live floor activity.
Columns: table_id, window_start, window_end, game_type, table_x, table_y, limit_min, limit_max, active_players (INT), bets (INT), avg_bet (FLOAT), total_bet (FLOAT), max_bet (FLOAT), min_bet (FLOAT), theo_win_window (FLOAT)

### mv_table_latest
Latest window's activity per table (subset of mv_table_activity). Use this for "right now" per-table questions.
Columns: same as mv_table_activity.

### mv_table_recommendations
Floor Plan view: LEFT JOIN of tables_dim with mv_table_latest, plus a business-rule layer. Cold tables (no activity in the latest window) still appear with zeros. One row per physical table.
Columns: table_id, game_type, table_x, table_y, limit_min, limit_max, active_players, bets, avg_bet, max_bet, total_bet, theo_win_window, window_start, action_type (VARCHAR: RAISE_LIMIT/LOWER_LIMIT/HOT/COLD/HOLD), suggested_limit_min (FLOAT), suggested_limit_max (FLOAT)

## Business Context
- **Archetypes:** casual (50% of players, low bets, mostly slots + light baccarat), regular (30%, baccarat-heavy mixed games), high_roller (12%, baccarat VIP + blackjack, the Macau core), emerging (8%, testing baccarat VIP while still playing the pit — the key group to watch).
- **Action types:** URGENT_RETENTION = churn > 45% + silver+ tier; VIP_UPGRADE_CANDIDATE = ML predicts high-roller trajectory; RETENTION_OFFER = churn > 38%; STANDARD_RECOMMENDATION = healthy player.
- **House edges (house advantage) by game:** Slots 7.50%, Baccarat 1.15% (blended banker/player/tie — the Macau hero game), Roulette 5.26%, Blackjack 0.75%, Poker 2.50%. These are the hardcoded per-game edges used to compute theo_win.
- **Theoretical Win (Theo Win)** = Σ(bet_amount × house_edge_of_that_game). Casino's expected profit from a player's wagering, independent of short-term luck — the single most important player-value metric in casino marketing. Baccarat has a low edge but enormous volume, which is why Macau floors are ~88% baccarat by GGR.
- **Effective house edge** = cumulative_theo_win ÷ cumulative_wagered. A player who plays more baccarat or blackjack has a lower effective edge (less profitable per dollar wagered) than one who plays mostly slots, even at the same wager volume.
- **Offer value (reinvestment)** scales with cumulative_theo_win: 40% for URGENT_RETENTION, 35% for VIP_UPGRADE_CANDIDATE, 25% for RETENTION_OFFER, 15% for STANDARD_RECOMMENDATION.
- **High roller similarity** weights: bet size 20%, baccarat pref 18% (dominant game-mix signal on an Asian floor), blackjack pref 8%, poker pref 6%, F&B spend 6%, hotel spend 6%, category diversity 6%, spend velocity 10%, cumulative_theo_win 20%.
- **Floor plan:** 36 tables in fixed positions on a Macau-style floor. Slots cluster on the left (8 penny + 8 standard). Blackjack pit in the middle (4 standard + 2 high-limit). **Baccarat is the hero: 8 standard tables + 2 VIP high-limit tables on the center-right**. Roulette (2 wheels) and Poker (2 tables) are minor — they do not drive theo on an Asian floor. Each table has (limit_min, limit_max). `mv_table_recommendations.action_type` = RAISE_LIMIT (packed + betting near ceiling), LOWER_LIMIT (cold + min above entry-level), HOT, COLD, or HOLD.

## Rules
1. ONLY generate SELECT queries. Never generate INSERT, UPDATE, DELETE, DROP, or any DDL.
2. Always use LIMIT to avoid returning too many rows (max 20 unless the user asks for more).
3. For decimal values, use `ROUND(value::numeric, n)` or `ROUND(CAST(value AS NUMERIC), n)` because RisingWave requires NUMERIC for the 2-argument ROUND function.
4. When the user asks about current "high rollers", "VIP candidates", "top candidates", or "unique candidates", query mv_high_roller_radar.
5. Use mv_player_high_roller_similarity only for historical, time-window, or trend questions about how similarity changes over time.
6. When ranking or listing players/candidates, ensure one row per player_id unless the user explicitly asks for window-level history.
7. When the user asks about "recommendations" or "churn" or "retention", query mv_actionable_recommendations.
8. For general player stats, use mv_player_features.
9. RisingWave uses PostgreSQL-compatible SQL syntax.
"""

HISTORY_CONTEXT_GUIDANCE = """
## Conversation Memory
You may receive a "Conversation memory" section containing earlier user questions, SQL queries, and result rows.

- Use that memory to resolve follow-up references like "these 5 players", "them", "those candidates", "same group", or "their metrics".
- If a prior result set includes explicit player IDs or other identifiers, preserve and reuse that same set in the follow-up query unless the user asks to change the scope.
- Prefer the most recent relevant result set when the user refers to "these" or "them".
"""

SYSTEM_PROMPT = f"""{SCHEMA_CONTEXT}

## How to respond
When the user asks a question:
1. First, write the SQL query you will execute (in a ```sql block).
2. Then I will execute it and give you the results.
3. Provide a clear, concise natural language answer based on the results.

If the question cannot be answered with the available data, explain why and suggest what they could ask instead.
"""

SQL_EXTRACT_PROMPT = f"""{SCHEMA_CONTEXT}
{HISTORY_CONTEXT_GUIDANCE}

You are a SQL generator. Given a user question, output ONLY a single SQL SELECT query (no markdown, no explanation, just raw SQL). The query must be read-only and include LIMIT. If the question cannot be answered, output: SELECT 'Cannot answer this question with available data' AS error;
"""


def extract_sql(text: str) -> str | None:
    """Extract SQL from LLM response (handles raw SQL or markdown code blocks)."""
    # Try to find SQL in code block
    match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Try plain code block
    match = re.search(r"```\s*(SELECT.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # If the whole response looks like SQL
    stripped = text.strip()
    if stripped.upper().startswith("SELECT"):
        return stripped.rstrip(";") + ";"
    return None


def _find_matching_paren(text: str, open_idx: int) -> int:
    """Return the index of the closing paren matching text[open_idx]."""
    depth = 0
    in_single = False
    in_double = False
    i = open_idx

    while i < len(text):
        ch = text[i]

        if ch == "'" and not in_double:
            if in_single and i + 1 < len(text) and text[i + 1] == "'":
                i += 2
                continue
            in_single = not in_single
        elif ch == '"' and not in_single:
            if in_double and i + 1 < len(text) and text[i + 1] == '"':
                i += 2
                continue
            in_double = not in_double
        elif not in_single and not in_double:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return i
        i += 1

    raise ValueError("Unbalanced parentheses in generated SQL.")


def _split_top_level_args(text: str) -> list[str]:
    """Split a function argument list on top-level commas only."""
    parts = []
    start = 0
    depth = 0
    in_single = False
    in_double = False
    i = 0

    while i < len(text):
        ch = text[i]

        if ch == "'" and not in_double:
            if in_single and i + 1 < len(text) and text[i + 1] == "'":
                i += 2
                continue
            in_single = not in_single
        elif ch == '"' and not in_single:
            if in_double and i + 1 < len(text) and text[i + 1] == '"':
                i += 2
                continue
            in_double = not in_double
        elif not in_single and not in_double:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 0:
                parts.append(text[start:i].strip())
                start = i + 1
        i += 1

    parts.append(text[start:].strip())
    return parts


def _is_numeric_round_arg(expr: str) -> bool:
    """Heuristic: whether an expression is already explicitly cast to NUMERIC/DECIMAL."""
    normalized = re.sub(r"\s+", " ", expr.strip()).upper()
    if "::NUMERIC" in normalized or "::DECIMAL" in normalized:
        return True
    if normalized.startswith("CAST(") and (" AS NUMERIC" in normalized or " AS DECIMAL" in normalized):
        return True
    return False


def _normalize_round_calls(sql: str) -> str:
    """Rewrite ROUND(x, n) to ROUND(CAST(x AS NUMERIC), n) when needed."""
    out = []
    i = 0

    while i < len(sql):
        match = re.search(r"\bROUND\s*\(", sql[i:], re.IGNORECASE)
        if not match:
            out.append(sql[i:])
            break

        start = i + match.start()
        paren_idx = start + match.group(0).rfind("(")
        out.append(sql[i:start])

        close_idx = _find_matching_paren(sql, paren_idx)
        inner = _normalize_round_calls(sql[paren_idx + 1:close_idx])
        args = _split_top_level_args(inner)

        if len(args) == 2 and not _is_numeric_round_arg(args[0]):
            args[0] = f"CAST({args[0].strip()} AS NUMERIC)"

        rebuilt = f"{sql[start:paren_idx]}({', '.join(args)})"
        out.append(rebuilt)
        i = close_idx + 1

    return "".join(out)


def normalize_generated_sql(sql: str) -> str:
    """Apply compatibility fixes to LLM-generated SQL before execution."""
    return _normalize_round_calls(sql.strip())


def _truncate(text: str, max_chars: int = 500) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _coerce_result_frame(data) -> pd.DataFrame | None:
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, list):
        return pd.DataFrame(data)
    return None


def build_history_context(history: list[dict] | None, max_messages: int = 6, max_rows: int = 8) -> str:
    """Render recent chat history into a compact, structured memory block for the LLM."""
    if not history:
        return ""

    blocks = []
    for idx, message in enumerate(history[-max_messages:], start=1):
        role = message.get("role", "assistant").upper()
        block_lines = [f"Memory item {idx} ({role})"]

        content = _truncate(message.get("content", ""))
        if content:
            block_lines.append(f"Message: {content}")

        sql = message.get("sql")
        if sql:
            block_lines.append(f"SQL: {sql.strip()}")

        df = _coerce_result_frame(message.get("data"))
        if df is not None and not df.empty:
            block_lines.append("Result rows:")
            block_lines.append(df.head(max_rows).to_markdown(index=False))

        blocks.append("\n".join(block_lines))

    return "\n\n".join(blocks)


def question_with_history(question: str, history: list[dict] | None) -> str:
    """Combine the current question with structured conversation memory."""
    history_context = build_history_context(history)
    if not history_context:
        return question
    return f"Conversation memory:\n{history_context}\n\nCurrent user question:\n{question}"


_CF_ERROR_TITLE = re.compile(r"<title>([^<]+)</title>", re.IGNORECASE)
_CF_H1 = re.compile(r"<h1[^>]*>.*?<span[^>]*>([^<]+)</span>.*?<span[^>]*>([^<]+)</span>", re.IGNORECASE | re.DOTALL)


def _clean_err(e: Exception) -> str:
    """Turn verbose upstream errors (especially HTML error pages from proxies) into a short line."""
    s = str(e)
    # If the body is an HTML page, extract the title + error code and drop the rest.
    if "<!DOCTYPE html" in s or "<html" in s.lower():
        title_m = _CF_ERROR_TITLE.search(s)
        h1_m = _CF_H1.search(s)
        parts = []
        if title_m:
            parts.append(title_m.group(1).strip())
        if h1_m:
            parts.append(f"{h1_m.group(1).strip()} ({h1_m.group(2).strip()})")
        if parts:
            return " — ".join(parts) + " — upstream proxy/API gateway error. Try a different model or retry."
        return "Upstream returned an HTML error page (likely a 5xx from the proxy). Try a different model or retry."
    # Keep length reasonable for any other verbose error
    if len(s) > 600:
        return s[:600] + "… [truncated]"
    return s


def execute_sql(conn, sql: str) -> tuple[pd.DataFrame | None, str | None]:
    """Execute a read-only SQL query and return results or error."""
    # Safety check
    upper = sql.upper().strip()
    if any(kw in upper for kw in ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"]):
        return None, "Blocked: only SELECT queries are allowed."
    try:
        df = pd.read_sql(sql, conn)
        return df, None
    except Exception as e:
        return None, str(e)


class ChatAgent(ABC):
    """Base class for LLM-powered chat agents."""

    @abstractmethod
    def generate_sql(self, question: str, history: list[dict] | None = None) -> str:
        """Given a user question, return a SQL query string."""
        ...

    @abstractmethod
    def summarize(
        self,
        question: str,
        sql: str,
        results: pd.DataFrame,
        history: list[dict] | None = None,
    ) -> str:
        """Given the question, SQL, and results, return a natural language answer."""
        ...

    def ask(self, question: str, conn, history: list[dict] | None = None) -> dict:
        """Full pipeline: question → SQL → execute → answer."""
        try:
            sql = self.generate_sql(question, history=history)
        except Exception as e:
            return {"error": f"LLM error: {_clean_err(e)}", "sql": None, "answer": None}

        if not sql:
            return {"error": "Could not generate SQL from the question.", "sql": None, "answer": None}

        sql = normalize_generated_sql(sql)

        df, err = execute_sql(conn, sql)
        if err:
            return {"error": err, "sql": sql, "answer": None}

        try:
            answer = self.summarize(question, sql, df, history=history)
        except Exception as e:
            # Fall back to raw data if summarization fails
            answer = f"(Summarization failed: {_clean_err(e)})\n\nQuery returned {len(df)} rows:\n\n{df.to_markdown(index=False)}"

        return {"error": None, "sql": sql, "answer": answer, "data": df}


class ClaudeAgent(ChatAgent):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514",
                 base_url: str | None = None, timeout: float = 60.0):
        import anthropic
        kwargs = {"api_key": api_key, "timeout": timeout, "max_retries": 1}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = anthropic.Anthropic(**kwargs)
        self.model = model

    def generate_sql(self, question: str, history: list[dict] | None = None) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system=SQL_EXTRACT_PROMPT,
            messages=[{"role": "user", "content": question_with_history(question, history)}],
        )
        raw = resp.content[0].text
        sql = extract_sql(raw)
        return sql or raw

    def summarize(
        self,
        question: str,
        sql: str,
        results: pd.DataFrame,
        history: list[dict] | None = None,
    ) -> str:
        data_str = results.head(20).to_markdown(index=False)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            system="You are a casino data analyst. Summarize the query results in a clear, concise answer. Use bullet points for lists. Include key numbers. Rewrite technical column names into plain English and avoid snake_case identifiers with underscores.",
            messages=[
                {"role": "user", "content": f"Question: {question}\n\nSQL executed:\n```sql\n{sql}\n```\n\nResults:\n{data_str}\n\nProvide a concise answer:"},
            ],
        )
        return resp.content[0].text


class OpenAIAgent(ChatAgent):
    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str | None = None,
                 timeout: float = 60.0, default_headers: dict | None = None):
        import openai
        kwargs = {"api_key": api_key, "timeout": timeout, "max_retries": 1}
        if base_url:
            kwargs["base_url"] = base_url
        if default_headers:
            kwargs["default_headers"] = default_headers
        self.client = openai.OpenAI(**kwargs)
        self.model = model

    def generate_sql(self, question: str, history: list[dict] | None = None) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=500,
            messages=[
                {"role": "system", "content": SQL_EXTRACT_PROMPT},
                {"role": "user", "content": question_with_history(question, history)},
            ],
        )
        raw = resp.choices[0].message.content
        sql = extract_sql(raw)
        return sql or raw

    def summarize(
        self,
        question: str,
        sql: str,
        results: pd.DataFrame,
        history: list[dict] | None = None,
    ) -> str:
        data_str = results.head(20).to_markdown(index=False)
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=800,
            messages=[
                {"role": "system", "content": "You are a casino data analyst. Summarize the query results in a clear, concise answer. Use bullet points for lists. Include key numbers. Rewrite technical column names into plain English and avoid snake_case identifiers with underscores."},
                {"role": "user", "content": f"Question: {question}\n\nSQL executed:\n```sql\n{sql}\n```\n\nResults:\n{data_str}\n\nProvide a concise answer:"},
            ],
        )
        return resp.choices[0].message.content


def create_agent(provider: str, api_key: str, **kwargs) -> ChatAgent:
    """Factory function to create the right agent based on provider selection."""
    if provider == "Claude":
        return ClaudeAgent(api_key=api_key, **kwargs)
    elif provider == "OpenAI":
        return OpenAIAgent(api_key=api_key, **kwargs)
    elif provider == "OpenRouter":
        # OpenRouter is OpenAI-compatible; set the base_url if not overridden and
        # attach the optional attribution headers OpenRouter recommends.
        kwargs.setdefault("base_url", "https://openrouter.ai/api/v1")
        kwargs.setdefault("model", "openai/gpt-4o")
        headers = kwargs.pop("default_headers", None) or {}
        headers.setdefault("HTTP-Referer", "http://localhost:8501")
        headers.setdefault("X-Title", "Smart Casino Floor Dashboard")
        return OpenAIAgent(api_key=api_key, default_headers=headers, **kwargs)
    elif provider == "Azure OpenAI":
        return OpenAIAgent(api_key=api_key, base_url=kwargs.get("base_url"), model=kwargs.get("model", "gpt-4o"))
    else:
        raise ValueError(f"Unknown provider: {provider}")
