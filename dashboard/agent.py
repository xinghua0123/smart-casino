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
Columns: player_id (VARCHAR), window_start (TIMESTAMPTZ), window_end (TIMESTAMPTZ), games_played (INT), avg_bet (FLOAT), total_bet (FLOAT), total_payout (FLOAT), win_rate (FLOAT 0-1), pct_slots (FLOAT 0-1), pct_roulette (FLOAT 0-1), pct_blackjack (FLOAT 0-1), pct_poker (FLOAT 0-1), cumulative_gaming_spend (FLOAT), tier (VARCHAR: bronze/silver/gold/platinum/diamond), archetype (VARCHAR: casual/regular/high_roller/emerging), fnb_orders (INT), fnb_spend (FLOAT), cumulative_fnb_spend (FLOAT), hotel_events (INT), hotel_spend (FLOAT), hotel_action_types (INT), category_diversity (INT 1-3), spend_per_minute (FLOAT)

### mv_player_high_roller_similarity
Per-player high-roller similarity score (0-1). Higher = more similar to known VIPs.
Columns: player_id, window_start, tier, archetype, avg_bet, cumulative_gaming_spend, cumulative_fnb_spend, spend_per_minute, category_diversity, high_roller_similarity (FLOAT 0-1)

### mv_high_roller_radar
Filtered view: only non-VIP players with high_roller_similarity > 0.4 (emerging VIP candidates).
Same columns as mv_player_high_roller_similarity.

### mv_actionable_recommendations
ML predictions joined with business rules. One row per player.
Columns: player_id, next_best_game (VARCHAR), churn_probability (FLOAT 0-1), offer_sensitivity (VARCHAR: free_play/fnb_voucher/hotel_upgrade/cashback), high_roller_score (FLOAT 0-1), high_roller_trajectory (BOOLEAN), tier, avg_bet, cumulative_gaming_spend, action_type (VARCHAR: STANDARD_RECOMMENDATION/RETENTION_OFFER/VIP_UPGRADE_CANDIDATE/URGENT_RETENTION), offer_value (FLOAT), recommendation_ts (TIMESTAMPTZ)

### recommendations_tbl
Raw ML predictions (updated every 10s by the inference service).
Columns: player_id (PK), next_best_game, churn_probability, offer_sensitivity, high_roller_score, high_roller_trajectory, ts

### mv_dashboard_stats
Aggregated dashboard KPIs (single row).
Columns: active_players (INT), avg_bet_all (FLOAT), avg_spend_per_min (FLOAT), total_wagered (FLOAT)

## Business Context
- **Archetypes:** casual (50% of players, low bets, mostly slots), regular (30%, mixed games), high_roller (12%, high bets, table games), emerging (8%, gradually increasing bets — the key group to watch)
- **Action types:** URGENT_RETENTION = churn > 45% + silver+ tier; VIP_UPGRADE_CANDIDATE = ML predicts high-roller trajectory; RETENTION_OFFER = churn > 38%; STANDARD_RECOMMENDATION = healthy player
- **High roller similarity** is computed from: bet size (25%), blackjack preference (15%), poker preference (15%), F&B spend (10%), hotel spend (10%), category diversity (10%), spend velocity (15%)

## Rules
1. ONLY generate SELECT queries. Never generate INSERT, UPDATE, DELETE, DROP, or any DDL.
2. Always use LIMIT to avoid returning too many rows (max 20 unless the user asks for more).
3. Use ROUND() for decimal values to keep output readable.
4. When the user asks about "high rollers" or "VIP candidates", query mv_high_roller_radar or mv_player_high_roller_similarity.
5. When the user asks about "recommendations" or "churn" or "retention", query mv_actionable_recommendations.
6. For general player stats, use mv_player_features.
7. RisingWave uses PostgreSQL-compatible SQL syntax.
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
    def generate_sql(self, question: str) -> str:
        """Given a user question, return a SQL query string."""
        ...

    @abstractmethod
    def summarize(self, question: str, sql: str, results: pd.DataFrame) -> str:
        """Given the question, SQL, and results, return a natural language answer."""
        ...

    def ask(self, question: str, conn) -> dict:
        """Full pipeline: question → SQL → execute → answer."""
        try:
            sql = self.generate_sql(question)
        except Exception as e:
            return {"error": f"LLM error: {_clean_err(e)}", "sql": None, "answer": None}

        if not sql:
            return {"error": "Could not generate SQL from the question.", "sql": None, "answer": None}

        df, err = execute_sql(conn, sql)
        if err:
            return {"error": err, "sql": sql, "answer": None}

        try:
            answer = self.summarize(question, sql, df)
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

    def generate_sql(self, question: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system=SQL_EXTRACT_PROMPT,
            messages=[{"role": "user", "content": question}],
        )
        raw = resp.content[0].text
        sql = extract_sql(raw)
        return sql or raw

    def summarize(self, question: str, sql: str, results: pd.DataFrame) -> str:
        data_str = results.head(20).to_markdown(index=False)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            system="You are a casino data analyst. Summarize the query results in a clear, concise answer. Use bullet points for lists. Include key numbers.",
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

    def generate_sql(self, question: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=500,
            messages=[
                {"role": "system", "content": SQL_EXTRACT_PROMPT},
                {"role": "user", "content": question},
            ],
        )
        raw = resp.choices[0].message.content
        sql = extract_sql(raw)
        return sql or raw

    def summarize(self, question: str, sql: str, results: pd.DataFrame) -> str:
        data_str = results.head(20).to_markdown(index=False)
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=800,
            messages=[
                {"role": "system", "content": "You are a casino data analyst. Summarize the query results in a clear, concise answer. Use bullet points for lists. Include key numbers."},
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
