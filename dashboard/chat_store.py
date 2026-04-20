"""
SQL-backed chat session storage for the dashboard.

Defaults to RisingWave, but the connection is configured via CHAT_STORE_* env vars
so the app can later switch to Postgres without changing the call sites.
"""

import json
import os

import pandas as pd
import psycopg2


def _json_default(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _execute_sql(cur, sql: str, params: tuple | None = None) -> None:
    """Execute SQL with literalized parameters for RisingWave/Postgres compatibility."""
    if params:
        sql = cur.mogrify(sql, params).decode()
    cur.execute(sql)


class SqlChatStore:
    def __init__(
        self,
        backend: str,
        host: str,
        port: str,
        user: str,
        password: str,
        dbname: str,
        sslmode: str | None = None,
    ):
        self.backend = backend
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.dbname = dbname
        self.sslmode = sslmode

    @classmethod
    def from_env(
        cls,
        default_host: str,
        default_port: str,
        default_user: str = "root",
        default_dbname: str = "dev",
    ) -> "SqlChatStore":
        return cls(
            backend=os.environ.get("CHAT_STORE_BACKEND", "risingwave"),
            host=os.environ.get("CHAT_STORE_HOST", default_host),
            port=os.environ.get("CHAT_STORE_PORT", default_port),
            user=os.environ.get("CHAT_STORE_USER", default_user),
            password=os.environ.get("CHAT_STORE_PASSWORD", ""),
            dbname=os.environ.get("CHAT_STORE_DBNAME", default_dbname),
            sslmode=(os.environ.get("CHAT_STORE_SSLMODE", "") or None),
        )

    def connect(self):
        kwargs = {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "dbname": self.dbname,
        }
        if self.password:
            kwargs["password"] = self.password
        if self.sslmode:
            kwargs["sslmode"] = self.sslmode
        conn = psycopg2.connect(**kwargs)
        conn.autocommit = True
        return conn

    def ensure_schema(self, conn) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    session_id VARCHAR,
                    message_index INTEGER,
                    role VARCHAR,
                    content VARCHAR,
                    sql_text VARCHAR,
                    result_json VARCHAR,
                    created_at TIMESTAMPTZ
                );
                """
            )

    def append_message(self, conn, session_id: str, message_index: int, message: dict) -> None:
        result_json = None
        data = message.get("data")
        if isinstance(data, pd.DataFrame) and not data.empty:
            result_json = json.dumps(data.to_dict(orient="records"), default=_json_default)

        with conn.cursor() as cur:
            # Idempotent write for reruns: keep one row per session + index.
            _execute_sql(
                cur,
                "DELETE FROM chat_messages WHERE session_id = %s AND message_index = %s",
                (session_id, message_index),
            )
            _execute_sql(
                cur,
                """
                INSERT INTO chat_messages (
                    session_id, message_index, role, content, sql_text, result_json, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                """,
                (
                    session_id,
                    message_index,
                    message.get("role"),
                    message.get("content"),
                    message.get("sql"),
                    result_json,
                ),
            )

    def load_messages(self, conn, session_id: str) -> list[dict]:
        with conn.cursor() as cur:
            _execute_sql(
                cur,
                """
                SELECT role, content, sql_text, result_json
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY message_index ASC, created_at ASC
                """,
                (session_id,),
            )
            rows = cur.fetchall()

        messages = []
        for role, content, sql_text, result_json in rows:
            message = {
                "role": role,
                "content": content,
            }
            if sql_text:
                message["sql"] = sql_text
            if result_json:
                try:
                    records = json.loads(result_json)
                    if records:
                        message["data"] = pd.DataFrame(records)
                except json.JSONDecodeError:
                    pass
            messages.append(message)
        return messages

    def clear_session(self, conn, session_id: str) -> None:
        with conn.cursor() as cur:
            _execute_sql(cur, "DELETE FROM chat_messages WHERE session_id = %s", (session_id,))
