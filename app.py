import io
import re
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DB_PATH = Path(__file__).with_name("banca.db")
VTEST_LABEL = "Vtest"


# ---------- Database ----------
def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                initial_bankroll REAL NOT NULL DEFAULT 0,
                days_in_month INTEGER NOT NULL DEFAULT 30,
                daily_loss_limit REAL NOT NULL DEFAULT 0,
                daily_profit_goal REAL NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                op_date TEXT NOT NULL,
                op_datetime TEXT,
                description TEXT NOT NULL,
                amount REAL NOT NULL,
                source TEXT NOT NULL DEFAULT 'manual',
                external_id TEXT,
                account_type TEXT NOT NULL DEFAULT 'MANUAL',
                symbol TEXT,
                operation_type TEXT,
                api_output TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        _ensure_column(conn, "operations", "op_datetime", "op_datetime TEXT")
        _ensure_column(conn, "operations", "source", "source TEXT NOT NULL DEFAULT 'manual'")
        _ensure_column(conn, "operations", "external_id", "external_id TEXT")
        _ensure_column(conn, "operations", "account_type", "account_type TEXT NOT NULL DEFAULT 'MANUAL'")
        _ensure_column(conn, "operations", "symbol", "symbol TEXT")
        _ensure_column(conn, "operations", "operation_type", "operation_type TEXT")
        _ensure_column(conn, "operations", "api_output", "api_output TEXT")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_operations_source_external_id ON operations(source, external_id)")
        conn.execute(
            """
            INSERT OR IGNORE INTO settings (id, initial_bankroll, days_in_month, daily_loss_limit, daily_profit_goal)
            VALUES (1, 0, 30, 0, 0)
            """
        )


def load_settings() -> dict[str, Any]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM settings WHERE id = 1").fetchone()
    return dict(row)


def save_settings(initial_bankroll: float, days_in_month: int, daily_loss_limit: float, daily_profit_goal: float) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE settings
            SET initial_bankroll = ?, days_in_month = ?, daily_loss_limit = ?, daily_profit_goal = ?
            WHERE id = 1
            """,
            (initial_bankroll, days_in_month, daily_loss_limit, daily_profit_goal),
        )


def add_operation(
    op_date: date,
    description: str,
    amount: float,
    source: str = "manual",
    external_id: str | None = None,
    op_datetime: datetime | None = None,
    account_type: str = "MANUAL",
    symbol: str | None = None,
    operation_type: str | None = None,
    api_output: str | None = None,
) -> bool:
    with get_connection() as conn:
        before = conn.total_changes
        conn.execute(
            """
            INSERT OR IGNORE INTO operations
            (op_date, op_datetime, description, amount, source, external_id, account_type, symbol, operation_type, api_output)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                op_date.isoformat(),
                op_datetime.isoformat(timespec="seconds") if op_datetime else None,
                description.strip(),
                amount,
                source,
                external_id,
                account_type,
                symbol,
                operation_type,
                api_output,
            ),
        )
        return conn.total_changes > before


def remove_operation(op_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM operations WHERE id = ?", (op_id,))


def load_operations() -> pd.DataFrame:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, op_date, op_datetime, description, amount, source, external_id, account_type, symbol, operation_type, api_output, created_at FROM operations ORDER BY op_date, id"
        ).fetchall()

    if not rows:
        return pd.DataFrame(columns=["id", "op_date", "op_datetime", "description", "amount", "source", "external_id", "account_type", "symbol", "operation_type", "api_output", "created_at"])
    return pd.DataFrame([dict(r) for r in rows])


# ---------- CSV / PDF Import ----------

# Mapping of known IQ Option CSV column aliases → normalised names
_COL_ALIASES: dict[str, str] = {
    # date/time
    "created": "datetime",
    "date": "datetime",
    "time": "datetime",
    "data": "datetime",
    "hora": "datetime",
    "data/hora": "datetime",
    "close_time": "datetime",
    "closed": "datetime",
    # asset / symbol
    "active": "symbol",
    "asset": "symbol",
    "ativo": "symbol",
    "instrumento": "symbol",
    "instrument": "symbol",
    # direction / type
    "direction": "operation_type",
    "type": "operation_type",
    "tipo": "operation_type",
    "kind": "operation_type",
    # invest / amount
    "amount": "invest",
    "invest": "invest",
    "valor": "invest",
    "investimento": "invest",
    "investment": "invest",
    # profit/loss result
    "profit": "profit",
    "profit/loss": "profit",
    "lucro": "profit",
    "resultado": "profit",
    "result": "profit",
    "revenue": "profit",
    "receita": "profit",
    "pnl": "profit",
    # status
    "status": "status",
    "win/loose": "status",  # IQ Option uses "loose" (typo in their export)
    "win/loss": "status",
    "win/loss": "status",
    # external id
    "id": "external_id",
    "order id": "external_id",
    "order_id": "external_id",
    "trade id": "external_id",
    "trade_id": "external_id",
    # account type
    "balance": "balance",
    "account": "account_type",
    "conta": "account_type",
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        key = col.strip().lower().replace("-", "/")
        if key in _COL_ALIASES:
            renamed[col] = _COL_ALIASES[key]
    df = df.rename(columns=renamed)
    # Keep only first occurrence when duplicate normalised names arise
    seen: set[str] = set()
    keep = []
    for c in df.columns:
        if c not in seen:
            keep.append(c)
            seen.add(c)
    return df[keep]


def _parse_float(val: Any) -> float | None:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace("R$", "").replace("$", "").replace("\xa0", "").replace(" ", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_datetime(val: Any) -> datetime | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, datetime):
        return val
    s = str(val).strip()
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
    ):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        return pd.to_datetime(s).to_pydatetime()
    except Exception:
        return None


def parse_iqoption_csv(file_bytes: bytes) -> list[dict[str, Any]]:
    """Parse an IQ Option CSV export and return a list of operation dicts."""
    try:
        text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1")

    # Try semicolon first (common IQ Option export), then comma
    for sep in (";", ",", "\t"):
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str)
            if df.shape[1] > 1:
                break
        except Exception:
            continue
    else:
        df = pd.read_csv(io.StringIO(text), dtype=str)

    df = _normalise_columns(df)
    operations: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        dt = _parse_datetime(row.get("datetime"))
        op_date = dt.date() if dt else date.today()

        invest = _parse_float(row.get("invest"))
        profit_raw = _parse_float(row.get("profit"))

        # Compute net P&L
        if profit_raw is not None and invest is not None:
            # IQ Option "profit" column can be either:
            #   a) gross revenue (invest + net_profit) — value is > invest on win
            #   b) net P&L directly (positive = win, negative = loss)
            # We detect case (a) when profit_raw > invest (gross payout > stake).
            if profit_raw > invest:
                amount = profit_raw - invest   # gross revenue → net P&L
            elif profit_raw < 0:
                amount = profit_raw            # already a net loss
            else:
                # profit_raw is between 0 and invest: treat as net P&L
                # (partial payout or direct net value)
                amount = profit_raw
        elif profit_raw is not None:
            amount = profit_raw
        else:
            status = str(row.get("status", "")).strip().lower()
            if status in ("win", "ganho", "ganhou", "lucro"):
                amount = invest if invest else 0.0
            elif status in ("loose", "loss", "perda", "perdeu"):  # "loose" is IQ Option's own typo
                amount = -invest if invest else 0.0
            else:
                continue  # not enough data

        symbol = str(row.get("symbol", "ATIVO")).strip() or "ATIVO"
        op_type = str(row.get("operation_type", "trade")).strip().lower() or "trade"
        ext_id = str(row.get("external_id", "")).strip()
        if not ext_id:
            ext_id = f"csv|{op_date}|{symbol}|{op_type}|{amount}"

        account_type = str(row.get("account_type", "REAL")).strip().upper() or "REAL"

        operations.append(
            {
                "op_date": op_date,
                "op_datetime": dt,
                "description": f"IQ Option | {op_type} | {symbol}",
                "amount": amount,
                "source": "iqoption_csv",
                "external_id": ext_id,
                "account_type": account_type,
                "symbol": symbol,
                "operation_type": op_type,
                "api_output": None,
            }
        )

    # Deduplicate
    seen: dict[str, dict[str, Any]] = {}
    for op in operations:
        seen[op["external_id"]] = op
    return list(seen.values())


def parse_iqoption_pdf(file_bytes: bytes) -> list[dict[str, Any]]:
    """Extract trades from an IQ Option PDF export using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError("Instale pdfplumber: pip install pdfplumber")

    rows: list[list[str]] = []
    header: list[str] | None = None

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table:
                    continue
                for i, row in enumerate(table):
                    if row is None:
                        continue
                    clean = [str(c).strip() if c is not None else "" for c in row]
                    if header is None:
                        # Detect header row
                        lower = [c.lower() for c in clean]
                        if any(k in lower for k in ("active", "ativo", "asset", "created", "date", "data")):
                            header = clean
                        continue
                    rows.append(clean)

    if header is None or not rows:
        return []

    df = pd.DataFrame(rows, columns=header, dtype=str)
    df = _normalise_columns(df)
    return parse_iqoption_csv(df.to_csv(index=False).encode())


# ---------- Analysis helpers ----------

def compute_trade_stats(operations: pd.DataFrame) -> dict[str, Any]:
    """Return a dict of aggregate statistics for the given operations."""
    if operations.empty:
        return {}

    df = operations.copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

    wins = df[df["amount"] > 0]
    losses = df[df["amount"] < 0]

    total = len(df)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / total * 100 if total else 0

    avg_win = float(wins["amount"].mean()) if not wins.empty else 0.0
    avg_loss = float(losses["amount"].mean()) if not losses.empty else 0.0
    profit_factor = (
        float(wins["amount"].sum() / abs(losses["amount"].sum()))
        if not losses.empty and losses["amount"].sum() != 0
        else float("inf")
    )

    # Consecutive wins/losses
    streaks = df["amount"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).tolist()
    max_consec_win = max_consec_loss = cur = 0
    for s in streaks:
        if s == 0:
            cur = 0
        elif (cur > 0 and s > 0) or (cur < 0 and s < 0):
            cur += s
        else:
            cur = s
        if cur > 0:
            max_consec_win = max(max_consec_win, cur)
        else:
            max_consec_loss = max(max_consec_loss, abs(cur))

    return {
        "total": total,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_consec_win": max_consec_win,
        "max_consec_loss": max_consec_loss,
        "gross_profit": float(wins["amount"].sum()),
        "gross_loss": float(abs(losses["amount"].sum())),
    }


def compute_drawdown(operations: pd.DataFrame, initial_bankroll: float) -> pd.DataFrame:
    """Return a DataFrame with cumulative balance and drawdown series."""
    if operations.empty:
        return pd.DataFrame(columns=["datetime", "balance", "drawdown_pct"])

    df = operations.copy()
    df["event_dt"] = pd.to_datetime(df["op_datetime"], errors="coerce")
    fallback = pd.to_datetime(df["op_date"], errors="coerce")
    df["event_dt"] = df["event_dt"].fillna(fallback)
    df = df.dropna(subset=["event_dt"]).sort_values("event_dt")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    df["balance"] = initial_bankroll + df["amount"].cumsum()
    df["peak"] = df["balance"].cummax()
    df["drawdown_pct"] = (df["balance"] - df["peak"]) / df["peak"].replace(0, float("nan")) * 100
    return df[["event_dt", "balance", "drawdown_pct", "peak"]].rename(columns={"event_dt": "datetime"})


# ---------- Risk Management ----------

def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Return the Kelly fraction (0–1) for position sizing."""
    if avg_loss == 0 or avg_win == 0:
        return 0.0
    b = abs(avg_win / avg_loss)
    p = win_rate / 100
    q = 1 - p
    kelly = (b * p - q) / b
    return max(0.0, min(kelly, 1.0))


def risk_recommendations(
    stats: dict[str, Any],
    settings: dict[str, Any],
    current_balance: float,
) -> list[dict[str, str]]:
    """Generate rule-based risk recommendations."""
    recs: list[dict[str, str]] = []

    win_rate = stats.get("win_rate", 0)
    avg_win = stats.get("avg_win", 0)
    avg_loss = stats.get("avg_loss", 0)
    profit_factor = stats.get("profit_factor", 0)
    max_consec_loss = stats.get("max_consec_loss", 0)

    kelly = kelly_criterion(win_rate, avg_win, abs(avg_loss))
    half_kelly = kelly / 2

    if half_kelly > 0:
        suggested_trade = current_balance * half_kelly
        recs.append(
            {
                "type": "info",
                "title": "📐 Tamanho de posição sugerido (½ Kelly)",
                "body": f"Com win rate de {win_rate:.1f}% e razão ganho/perda de {abs(avg_win/avg_loss):.2f}x, "
                        f"o critério de Kelly sugere arriscar **{half_kelly*100:.1f}%** da banca por operação "
                        f"≈ **R$ {suggested_trade:,.2f}**.",
            }
        )

    if win_rate < 50:
        recs.append(
            {
                "type": "warning",
                "title": "⚠️ Win rate abaixo de 50%",
                "body": f"Seu win rate atual é {win_rate:.1f}%. Considere revisar a estratégia ou reduzir o tamanho das posições até melhorar a assertividade.",
            }
        )
    elif win_rate >= 65:
        recs.append(
            {
                "type": "success",
                "title": "✅ Win rate saudável",
                "body": f"Win rate de {win_rate:.1f}% é excelente. Mantenha a disciplina operacional.",
            }
        )

    if profit_factor < 1.0:
        recs.append(
            {
                "type": "error",
                "title": "🔴 Profit factor negativo",
                "body": f"Profit factor de {profit_factor:.2f} significa que você está perdendo mais do que ganhando. Pare e revise a estratégia.",
            }
        )
    elif profit_factor >= 1.5:
        recs.append(
            {
                "type": "success",
                "title": "✅ Profit factor positivo",
                "body": f"Profit factor de {profit_factor:.2f} indica estratégia lucrativa. Continue aplicando a gestão de risco.",
            }
        )

    if max_consec_loss >= 5:
        recs.append(
            {
                "type": "error",
                "title": "🔴 Sequência de perdas alta",
                "body": f"Você já teve até {max_consec_loss} perdas consecutivas. Defina uma regra de pausa após 3–4 perdas seguidas para proteger a banca.",
            }
        )

    daily_loss_pct = float(settings.get("daily_loss_limit", 0))
    if daily_loss_pct == 0:
        recs.append(
            {
                "type": "warning",
                "title": "⚠️ Stop loss diário não configurado",
                "body": "Configure um limite de perda diária nas configurações (recomendado: 5–10% da banca) para proteger seu capital.",
            }
        )

    return recs


# ---------- AI Suggestions ----------

def ai_suggestions(operations: pd.DataFrame, stats: dict[str, Any]) -> list[dict[str, str]]:
    """Rule-based intelligence suggestions derived from trade history."""
    suggestions: list[dict[str, str]] = []

    if operations.empty or not stats:
        suggestions.append(
            {
                "type": "info",
                "title": "📂 Importe seus trades",
                "body": "Carregue um arquivo CSV ou PDF da IQ Option para receber sugestões personalizadas com base no seu histórico.",
            }
        )
        return suggestions

    df = operations.copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    df["event_dt"] = pd.to_datetime(df["op_datetime"], errors="coerce").fillna(
        pd.to_datetime(df["op_date"], errors="coerce")
    )

    # --- Best trading hour ---
    if df["event_dt"].notna().any():
        df["hour"] = df["event_dt"].dt.hour
        hour_stats = df.groupby("hour")["amount"].agg(["sum", "count", "mean"])
        if not hour_stats.empty:
            best_hour = int(hour_stats["mean"].idxmax())
            worst_hour = int(hour_stats["mean"].idxmin())
            best_mean = float(hour_stats.loc[best_hour, "mean"])
            worst_mean = float(hour_stats.loc[worst_hour, "mean"])
            if best_mean > 0:
                suggestions.append(
                    {
                        "type": "success",
                        "title": f"⏰ Melhor horário: {best_hour:02d}h00",
                        "body": f"Sua média de resultado entre {best_hour:02d}h e {best_hour+1:02d}h é de "
                                f"R$ {best_mean:+.2f} por trade. Concentre operações nesse período.",
                    }
                )
            if worst_mean < 0:
                suggestions.append(
                    {
                        "type": "warning",
                        "title": f"🚫 Evite operar às {worst_hour:02d}h00",
                        "body": f"Sua média de resultado entre {worst_hour:02d}h e {worst_hour+1:02d}h é de "
                                f"R$ {worst_mean:+.2f} por trade — o pior horário do seu histórico.",
                    }
                )

    # --- Best day of week ---
    if df["event_dt"].notna().any():
        day_names = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
        df["weekday"] = df["event_dt"].dt.weekday
        dow_stats = df.groupby("weekday")["amount"].mean()
        if not dow_stats.empty:
            best_day = int(dow_stats.idxmax())
            worst_day = int(dow_stats.idxmin())
            if dow_stats[best_day] > 0:
                suggestions.append(
                    {
                        "type": "success",
                        "title": f"📅 Melhor dia: {day_names[best_day]}",
                        "body": f"Você tem melhor desempenho médio às {day_names[best_day]}s "
                                f"(R$ {dow_stats[best_day]:+.2f}/trade). Priorize esse dia.",
                    }
                )
            if dow_stats[worst_day] < 0 and worst_day != best_day:
                suggestions.append(
                    {
                        "type": "warning",
                        "title": f"📅 Cuidado às {day_names[worst_day]}s",
                        "body": f"Seu pior desempenho médio ocorre às {day_names[worst_day]}s "
                                f"(R$ {dow_stats[worst_day]:+.2f}/trade). Considere operar menos nesse dia.",
                    }
                )

    # --- Best asset ---
    if "symbol" in df.columns and df["symbol"].notna().any():
        asset_stats = df.groupby("symbol")["amount"].agg(["sum", "count", "mean"])
        asset_stats = asset_stats[asset_stats["count"] >= 3]
        if not asset_stats.empty:
            best_asset = str(asset_stats["mean"].idxmax())
            worst_asset = str(asset_stats["mean"].idxmin())
            if float(asset_stats.loc[best_asset, "mean"]) > 0:
                suggestions.append(
                    {
                        "type": "success",
                        "title": f"🏆 Ativo mais lucrativo: {best_asset}",
                        "body": f"Você tem média de R$ {float(asset_stats.loc[best_asset, 'mean']):+.2f}/trade em {best_asset} "
                                f"({int(asset_stats.loc[best_asset, 'count'])} trades). Foque nesse ativo.",
                    }
                )
            if float(asset_stats.loc[worst_asset, "mean"]) < 0 and worst_asset != best_asset:
                suggestions.append(
                    {
                        "type": "warning",
                        "title": f"❌ Ativo menos lucrativo: {worst_asset}",
                        "body": f"Você perde em média R$ {abs(float(asset_stats.loc[worst_asset, 'mean'])):,.2f}/trade em {worst_asset}. "
                                f"Evite ou revise a estratégia para esse ativo.",
                    }
                )

    # --- Consecutive loss alert ---
    recent = df.sort_values("event_dt").tail(10)
    recent_streak = 0
    for amt in reversed(recent["amount"].tolist()):
        if amt < 0:
            recent_streak += 1
        else:
            break
    if recent_streak >= 3:
        suggestions.append(
            {
                "type": "error",
                "title": f"🔴 {recent_streak} perdas consecutivas recentes",
                "body": "Você teve perdas seguidas nos últimos trades. Pausar e revisar o setup antes de continuar é recomendado.",
            }
        )

    # --- Recovery suggestion after stop ---
    win_rate = stats.get("win_rate", 0)
    if 45 <= win_rate < 55:
        suggestions.append(
            {
                "type": "info",
                "title": "🎯 Win rate na zona neutra",
                "body": f"Com {win_rate:.1f}% de win rate, o resultado depende fortemente da relação ganho/perda por trade. "
                        f"Tente aumentar o payout médio ou reduzir o investimento médio em perdas.",
            }
        )

    if not suggestions:
        suggestions.append(
            {
                "type": "success",
                "title": "✅ Estratégia aparentemente sólida",
                "body": "Nenhuma anomalia significativa detectada no seu histórico. Mantenha a disciplina e o controle emocional.",
            }
        )

    return suggestions


# ---------- UI ----------
def _format_profit_factor(pf: float) -> str:
    """Format profit factor; returns '∞' for infinity."""
    if pf == float("inf"):
        return "∞"
    return f"{pf:.2f}"


def _gain_loss_ratio(avg_win: float, avg_loss: float) -> str:
    """Return formatted gain/loss ratio, guarding against zero denominator."""
    if avg_loss == 0.0:
        return "N/A"
    return f"{abs(avg_win / avg_loss):.2f}x"
def format_currency(value: float) -> str:
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def maybe_mask_currency(value: float, mask: bool) -> str:
    return "***" if mask else format_currency(value)


def signed_currency(value: float) -> str:
    signal = "+" if value >= 0 else "-"
    return f"{signal} {format_currency(abs(value))}"


def render_daily_available_panel(
    available_value: float,
    daily_pct: float,
    day_result: float,
    stop_loss_hit: bool,
    goal_hit: bool,
) -> None:
    panel_bg = "linear-gradient(135deg, #14532d, #166534)"
    panel_border = "#22c55e"
    glow = "0 0 18px rgba(34, 197, 94, 0.60)"
    status = "Positivo"

    if stop_loss_hit:
        panel_bg = "linear-gradient(135deg, #2b0a0a, #111111)"
        panel_border = "#7f1d1d"
        glow = "0 0 18px rgba(127, 29, 29, 0.80)"
        status = "STOP LOSS ATINGIDO"
    elif goal_hit:
        panel_bg = "linear-gradient(135deg, #7a5d00, #d4af37)"
        panel_border = "#facc15"
        glow = "0 0 20px rgba(250, 204, 21, 0.80)"
        status = "META DIÁRIA ATINGIDA"
    elif day_result < 0:
        panel_bg = "linear-gradient(135deg, #7f1d1d, #991b1b)"
        panel_border = "#ef4444"
        glow = "0 0 18px rgba(239, 68, 68, 0.65)"
        status = "Negativo"

    indicator_color = "#22c55e" if daily_pct >= 0 else "#ef4444"
    pct_signal = "+" if daily_pct >= 0 else ""

    st.markdown(
        f"""
        <div style="
            border: 1px solid {panel_border};
            border-radius: 16px;
            padding: 16px;
            margin: 8px 0 16px 0;
            background: {panel_bg};
            box-shadow: {glow};
            color: #f8fafc;
        ">
            <div style="font-size: 0.9rem; opacity: 0.9;">Painel do Dia</div>
            <div style="font-size: 1.7rem; font-weight: 700; margin-top: 4px;">Saldo disponível (dia): {format_currency(available_value)}</div>
            <div style="margin-top: 8px; font-size: 1rem;">
                Resultado do dia: <b>{signed_currency(day_result)}</b>
                &nbsp;|&nbsp;
                % do dia: <b style="color: {indicator_color};">{pct_signal}{daily_pct:.2f}%</b>
            </div>
            <div style="margin-top: 8px; font-size: 0.9rem; opacity: 0.95;">Status: <b>{status}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_percentage_panel(title: str, pct_value: float, result_value: float, hide_values: bool) -> None:
    positive = pct_value >= 0
    bg = "linear-gradient(135deg, #14532d, #166534)" if positive else "linear-gradient(135deg, #7f1d1d, #991b1b)"
    border = "#22c55e" if positive else "#ef4444"
    glow = "0 0 16px rgba(34, 197, 94, 0.60)" if positive else "0 0 16px rgba(239, 68, 68, 0.60)"
    pct_text = "***" if hide_values else f"{pct_value:+.2f}%"
    result_text = "***" if hide_values else signed_currency(result_value)

    st.markdown(
        f"""
        <div style="
            border: 1px solid {border};
            border-radius: 14px;
            padding: 12px;
            background: {bg};
            box-shadow: {glow};
            color: #f8fafc;
            min-height: 110px;
        ">
            <div style="font-size: 0.9rem; opacity: 0.9;">{title}</div>
            <div style="font-size: 1.5rem; font-weight: 700; margin-top: 4px;">{pct_text}</div>
            <div style="font-size: 0.95rem; margin-top: 6px;">Resultado: <b>{result_text}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_alert(rec: dict[str, str]) -> None:
    t = rec.get("type", "info")
    title = rec.get("title", "")
    body = rec.get("body", "")
    if t == "success":
        st.success(f"**{title}**  \n{body}")
    elif t == "warning":
        st.warning(f"**{title}**  \n{body}")
    elif t == "error":
        st.error(f"**{title}**  \n{body}")
    else:
        st.info(f"**{title}**  \n{body}")


def build_detailed_chart(operations: pd.DataFrame, initial_bankroll: float, group_by: str):
    df = operations.copy()
    df["event_dt"] = pd.to_datetime(df["op_datetime"], errors="coerce")
    fallback_dt = pd.to_datetime(df["op_date"], errors="coerce")
    df["event_dt"] = df["event_dt"].fillna(fallback_dt)
    df = df.dropna(subset=["event_dt"]).sort_values("event_dt")

    freq_map = {"Hora": "h", "Dia": "D", "Semana": "W", "Mês": "ME", "Ano": "YE"}
    freq = freq_map[group_by]
    grouped = (
        df.set_index("event_dt")
        .resample(freq)["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "pnl_periodo"})
    )
    grouped["saldo_acumulado"] = float(initial_bankroll) + grouped["pnl_periodo"].cumsum()
    grouped["resultado"] = grouped["pnl_periodo"].apply(lambda x: "Lucro" if x >= 0 else "Perda")

    fig = px.bar(
        grouped,
        x="event_dt",
        y="pnl_periodo",
        color="resultado",
        color_discrete_map={"Lucro": "#22c55e", "Perda": "#ef4444"},
        title=f"Desempenho por {group_by.lower()}",
        labels={"event_dt": "Período", "pnl_periodo": "Resultado"},
    )
    fig.add_scatter(
        x=grouped["event_dt"],
        y=grouped["saldo_acumulado"],
        mode="lines+markers",
        name="Saldo acumulado",
        line={"color": "#2563eb", "width": 3},
        yaxis="y2",
    )
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        legend_title_text="Métrica",
        yaxis={"title": "P&L do período"},
        yaxis2={"title": "Saldo acumulado", "overlaying": "y", "side": "right"},
        xaxis={"title": "Período"},
        bargap=0.2,
    )
    return fig, grouped


def main() -> None:
    st.set_page_config(page_title="Controle de Banca IQ Option", page_icon="💰", layout="wide")
    init_db()

    if "hide_values" not in st.session_state:
        st.session_state.hide_values = False

    settings = load_settings()
    operations = load_operations()

    total_gain = float(operations.loc[operations["amount"] > 0, "amount"].sum()) if not operations.empty else 0.0
    total_loss = float(abs(operations.loc[operations["amount"] < 0, "amount"].sum())) if not operations.empty else 0.0
    net_balance = total_gain - total_loss
    current_balance = float(settings["initial_bankroll"]) + net_balance
    days_in_month = max(int(settings["days_in_month"]), 1)
    today_iso = date.today().isoformat()
    yesterday_iso = (date.today() - pd.Timedelta(days=1)).isoformat()

    # Métricas congeladas no início do dia:
    # - consideram apenas operações anteriores à data atual
    # - não variam com novas operações lançadas no dia corrente
    historical_until_yesterday = operations[operations["op_date"] < today_iso] if not operations.empty else pd.DataFrame(columns=["amount"])
    net_until_yesterday = float(historical_until_yesterday["amount"].sum()) if not historical_until_yesterday.empty else 0.0

    daily_ops_today = operations[operations["op_date"] == today_iso] if not operations.empty else pd.DataFrame(columns=["amount"])
    daily_ops_yesterday = operations[operations["op_date"] == yesterday_iso] if not operations.empty else pd.DataFrame(columns=["amount"])
    daily_result_today = float(daily_ops_today["amount"].sum()) if not daily_ops_today.empty else 0.0
    daily_result_yesterday = float(daily_ops_yesterday["amount"].sum()) if not daily_ops_yesterday.empty else 0.0
    daily_gain_today = float(daily_ops_today.loc[daily_ops_today["amount"] > 0, "amount"].sum()) if not daily_ops_today.empty else 0.0
    daily_loss_today = float(abs(daily_ops_today.loc[daily_ops_today["amount"] < 0, "amount"].sum())) if not daily_ops_today.empty else 0.0
    daily_available_initial_adjusted = (float(settings["initial_bankroll"]) / days_in_month) + daily_result_yesterday
    daily_available_live = daily_available_initial_adjusted + daily_result_today
    daily_profit_pct = (daily_result_today / daily_available_initial_adjusted * 100) if daily_available_initial_adjusted != 0 else 0.0
    day_loss_limit_value = abs(daily_available_initial_adjusted) * (float(settings["daily_loss_limit"]) / 100)
    day_profit_goal_value = abs(daily_available_initial_adjusted) * (float(settings["daily_profit_goal"]) / 100)
    stop_loss_hit_today = float(settings["daily_loss_limit"]) > 0 and daily_result_today <= -day_loss_limit_value
    goal_hit_today = float(settings["daily_profit_goal"]) > 0 and daily_result_today >= day_profit_goal_value

    today_ts = pd.Timestamp(date.today())
    week_start = today_ts - pd.Timedelta(days=today_ts.weekday())
    month_start = today_ts.replace(day=1)
    year_start = today_ts.replace(month=1, day=1)
    op_dates = pd.to_datetime(operations["op_date"], errors="coerce") if not operations.empty else pd.Series(dtype="datetime64[ns]")

    week_ops = operations[op_dates >= week_start] if not operations.empty else pd.DataFrame(columns=["amount"])
    month_ops = operations[op_dates >= month_start] if not operations.empty else pd.DataFrame(columns=["amount"])
    year_ops = operations[op_dates >= year_start] if not operations.empty else pd.DataFrame(columns=["amount"])
    week_result = float(week_ops["amount"].sum()) if not week_ops.empty else 0.0
    month_result = float(month_ops["amount"].sum()) if not month_ops.empty else 0.0
    year_result = float(year_ops["amount"].sum()) if not year_ops.empty else 0.0

    base_week = float(settings["initial_bankroll"]) + (
        float(operations[op_dates < week_start]["amount"].sum()) if not operations.empty else 0.0
    )
    base_month = float(settings["initial_bankroll"]) + (
        float(operations[op_dates < month_start]["amount"].sum()) if not operations.empty else 0.0
    )
    base_year = float(settings["initial_bankroll"]) + (
        float(operations[op_dates < year_start]["amount"].sum()) if not operations.empty else 0.0
    )
    week_profit_pct = (week_result / base_week * 100) if base_week != 0 else 0.0
    month_profit_pct = (month_result / base_month * 100) if base_month != 0 else 0.0
    year_profit_pct = (year_result / base_year * 100) if base_year != 0 else 0.0

    # ---- Sidebar ----
    with st.sidebar:
        st.header("⚙️ Configurações")
        with st.form("settings_form"):
            initial_bankroll = st.number_input("Banca inicial", min_value=0.0, step=10.0, value=float(settings["initial_bankroll"]))
            days_month = st.number_input("Número de dias do mês", min_value=1, max_value=31, step=1, value=int(settings["days_in_month"]))
            daily_loss_limit = st.number_input(
                "Limite de perda diária (stop loss) %", min_value=0.0, max_value=100.0, step=0.1, value=float(settings["daily_loss_limit"])
            )
            daily_profit_goal = st.number_input(
                "Meta de lucro diária %", min_value=0.0, max_value=100.0, step=0.1, value=float(settings["daily_profit_goal"])
            )
            save = st.form_submit_button("Salvar configurações")

        if save:
            save_settings(initial_bankroll, int(days_month), daily_loss_limit, daily_profit_goal)
            st.success("Configurações salvas com sucesso.")
            st.rerun()

    st.title("💰 Controle de Banca IQ Option")
    st.caption(f"Gerencie sua banca, importe histórico da IQ Option e receba análises inteligentes. {VTEST_LABEL}")

    # ---- Main tabs ----
    tab_dashboard, tab_import, tab_analysis, tab_risk, tab_ai = st.tabs(
        ["📊 Dashboard", "📤 Importar Dados", "📈 Análise de Trades", "🛡️ Gestão de Risco", "🤖 IA & Sugestões"]
    )

    # ================================================================
    # TAB 1 — DASHBOARD
    # ================================================================
    with tab_dashboard:
        hide_values = bool(st.session_state.hide_values)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Saldo atualizado", maybe_mask_currency(current_balance, hide_values))
        if col1.button("👁️/🙈 Mostrar ou ocultar saldo", key="toggle_visibility"):
            st.session_state.hide_values = not hide_values
            st.rerun()
        col2.metric(
            "Disponível/dia (base inicial no início do dia)",
            maybe_mask_currency(daily_available_initial_adjusted, hide_values),
        )
        with col3:
            render_percentage_panel("Lucro do dia (%)", daily_profit_pct, daily_result_today, hide_values)
        col4.metric("Saldo líquido total", maybe_mask_currency(net_balance, hide_values))

        if hide_values:
            st.info("Painel do dia oculto porque a visualização de valores sensíveis está desativada.")
        else:
            render_daily_available_panel(
                available_value=daily_available_live,
                daily_pct=daily_profit_pct,
                day_result=daily_result_today,
                stop_loss_hit=stop_loss_hit_today,
                goal_hit=goal_hit_today,
            )

        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        dcol1.metric("Total ganho (dia)", maybe_mask_currency(daily_gain_today, hide_values))
        dcol2.metric("Total prejuízo (dia)", maybe_mask_currency(daily_loss_today, hide_values))
        dcol3.metric("Total ganho (acumulado)", maybe_mask_currency(total_gain, hide_values))
        dcol4.metric("Total perdido (acumulado)", maybe_mask_currency(total_loss, hide_values))

        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            render_percentage_panel("% lucro da semana", week_profit_pct, week_result, hide_values)
        with pcol2:
            render_percentage_panel("% lucro do mês", month_profit_pct, month_result, hide_values)
        with pcol3:
            render_percentage_panel("% lucro do ano", year_profit_pct, year_result, hide_values)

        st.divider()
        st.subheader("Registrar operação manual")
        with st.form("add_operation_form"):
            fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns([1, 1, 2, 1, 1])
            op_date = fcol1.date_input("Data", value=date.today())
            op_time = fcol2.time_input("Hora", value=datetime.now().time())
            description = fcol3.text_input("Descrição da operação", placeholder="Ex.: Trade EUR/USD")
            operation_type = fcol4.selectbox("Tipo", options=["Lucro", "Perda"])
            amount_input = fcol5.number_input("Valor", min_value=0.0, step=10.0)
            add_btn = st.form_submit_button("Adicionar operação")

        if add_btn:
            if not description.strip():
                st.error("Informe a descrição da operação.")
            elif amount_input <= 0:
                st.error("Informe um valor maior que zero.")
            else:
                signed_amount = amount_input if operation_type == "Lucro" else -amount_input
                add_operation(op_date, description, signed_amount, op_datetime=datetime.combine(op_date, op_time))
                st.success("Operação adicionada.")
                st.rerun()

        st.divider()
        st.subheader("Histórico de operações")
        if operations.empty:
            st.info("Nenhuma operação registrada ainda.")
        else:
            display_df = operations.copy()
            display_df["Data/Hora"] = pd.to_datetime(display_df["op_datetime"], errors="coerce").dt.strftime("%d/%m/%Y %H:%M")
            display_df["Data/Hora"] = display_df["Data/Hora"].fillna(pd.to_datetime(display_df["op_date"]).dt.strftime("%d/%m/%Y"))
            display_df["Valor"] = display_df["amount"].apply(format_currency)
            display_df = display_df.rename(
                columns={"id": "ID", "description": "Descrição", "source": "Origem", "account_type": "Conta", "symbol": "Ativo", "operation_type": "Tipo", "external_id": "ID Externo"}
            )
            st.dataframe(display_df[[c for c in ["ID", "Data/Hora", "Descrição", "Conta", "Ativo", "Tipo", "Origem", "ID Externo", "Valor"] if c in display_df.columns]], width="stretch", hide_index=True)

            remove_col1, remove_col2 = st.columns([2, 1])
            op_to_remove = remove_col1.selectbox(
                "Selecione o ID da operação para remover",
                options=list(operations["id"]),
                format_func=lambda x: f"ID {x} - {operations.loc[operations['id'] == x, 'description'].iloc[0]}",
            )
            if remove_col2.button("Remover operação", type="secondary"):
                remove_operation(int(op_to_remove))
                st.warning("Operação removida.")
                st.rerun()

        st.divider()
        st.subheader("Resumo líquido")
        st.write(f"**Saldo líquido:** {maybe_mask_currency(net_balance, hide_values)}")

        selected_date = st.date_input("Verificar alertas do dia", value=date.today(), key="alert_date")
        daily_result = 0.0
        if not operations.empty:
            day_ops = operations[operations["op_date"] == selected_date.isoformat()]
            daily_result = float(day_ops["amount"].sum()) if not day_ops.empty else 0.0

        st.write(f"Resultado do dia {selected_date.strftime('%d/%m/%Y')}: **{maybe_mask_currency(daily_result, hide_values)}**")
        st.caption(
            f"Limites do dia (base no saldo atual): stop loss = {format_currency(day_loss_limit_value)} "
            f"({settings['daily_loss_limit']:.2f}%), meta = {format_currency(day_profit_goal_value)} "
            f"({settings['daily_profit_goal']:.2f}%)."
        )
        if settings["daily_loss_limit"] > 0 and daily_result <= -day_loss_limit_value:
            st.error("⚠️ Stop loss diário atingido!")
        if settings["daily_profit_goal"] > 0 and daily_result >= day_profit_goal_value:
            st.success("🎯 Meta de lucro diária atingida!")

    # ================================================================
    # TAB 2 — IMPORTAR DADOS (CSV / PDF)
    # ================================================================
    with tab_import:
        st.header("📤 Importar Histórico da IQ Option")
        st.markdown(
            "Faça o upload do arquivo exportado diretamente da plataforma IQ Option "
            "(CSV ou PDF). As operações serão importadas automaticamente para o sistema."
        )

        with st.expander("ℹ️ Como exportar o histórico da IQ Option", expanded=False):
            st.markdown(
                """
                **Passos para exportar o histórico:**
                1. Acesse sua conta na IQ Option pelo navegador.
                2. Vá em **Histórico de Negociações** (ícone de relógio no menu lateral).
                3. Filtre o período desejado.
                4. Clique em **Exportar** e escolha **CSV** ou **PDF**.
                5. Faça o upload do arquivo aqui.

                > **Formatos suportados:** `.csv`, `.pdf`
                """
            )

        uploaded_file = st.file_uploader(
            "Selecione o arquivo CSV ou PDF exportado da IQ Option",
            type=["csv", "pdf"],
            help="Arraste ou clique para selecionar o arquivo exportado da IQ Option.",
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            file_name = uploaded_file.name.lower()

            with st.spinner("Processando arquivo..."):
                try:
                    if file_name.endswith(".pdf"):
                        parsed_ops = parse_iqoption_pdf(file_bytes)
                    else:
                        parsed_ops = parse_iqoption_csv(file_bytes)
                except Exception as exc:
                    st.error(f"Erro ao processar arquivo: {exc}")
                    parsed_ops = []

            if not parsed_ops:
                st.warning(
                    "Nenhuma operação encontrada no arquivo. "
                    "Verifique se o arquivo foi exportado corretamente da IQ Option."
                )
            else:
                st.success(f"**{len(parsed_ops)} operações** encontradas no arquivo.")

                preview_df = pd.DataFrame(parsed_ops).copy()
                preview_df["Valor"] = preview_df["amount"].apply(format_currency)
                preview_df["Data/Hora"] = pd.to_datetime(preview_df["op_datetime"], errors="coerce").dt.strftime("%d/%m/%Y %H:%M")
                preview_df["Data/Hora"] = preview_df["Data/Hora"].fillna(
                    pd.to_datetime(preview_df["op_date"].astype(str), errors="coerce").dt.strftime("%d/%m/%Y")
                )
                st.subheader("Pré-visualização das operações")
                show_cols = [c for c in ["Data/Hora", "symbol", "operation_type", "Valor", "account_type", "source"] if c in preview_df.columns]
                st.dataframe(preview_df[show_cols].rename(columns={"symbol": "Ativo", "operation_type": "Tipo", "account_type": "Conta", "source": "Origem"}), width="stretch", hide_index=True)

                if st.button("✅ Confirmar importação", type="primary"):
                    inserted = 0
                    skipped = 0
                    for op in parsed_ops:
                        did_insert = add_operation(
                            op["op_date"],
                            op["description"],
                            float(op["amount"]),
                            op.get("source", "iqoption_csv"),
                            op.get("external_id"),
                            op.get("op_datetime"),
                            op.get("account_type", "REAL"),
                            op.get("symbol"),
                            op.get("operation_type"),
                            op.get("api_output"),
                        )
                        if did_insert:
                            inserted += 1
                        else:
                            skipped += 1
                    st.success(f"Importação concluída! ✅ {inserted} novas operações inseridas, {skipped} já existiam.")
                    st.rerun()

    # ================================================================
    # TAB 3 — ANÁLISE DE TRADES
    # ================================================================
    with tab_analysis:
        st.header("📈 Análise de Trades")

        if operations.empty:
            st.info("Nenhuma operação encontrada. Importe seu histórico na aba **📤 Importar Dados**.")
        else:
            stats = compute_trade_stats(operations)

            # KPIs
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Total de trades", stats.get("total", 0))
            k2.metric("Win rate", f"{stats.get('win_rate', 0):.1f}%")
            k3.metric("Profit factor", _format_profit_factor(stats.get("profit_factor", 0)))
            k4.metric("Média de ganho", format_currency(stats.get("avg_win", 0)))
            k5.metric("Média de perda", format_currency(abs(stats.get("avg_loss", 0))))

            st.divider()

            # Performance chart (existing)
            st.subheader("📈 Desempenho detalhado das operações")
            timeframe = st.selectbox("Visualizar por", ["Hora", "Dia", "Semana", "Mês", "Ano"], index=1, key="analysis_timeframe")
            fig, grouped = build_detailed_chart(operations, float(settings["initial_bankroll"]), timeframe)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(
                grouped.rename(columns={"event_dt": "Período", "pnl_periodo": "P&L do período", "saldo_acumulado": "Saldo acumulado", "resultado": "Resultado"}),
                width="stretch",
                hide_index=True,
            )

            st.divider()

            acol1, acol2 = st.columns(2)

            # Win rate by asset
            with acol1:
                st.subheader("🏆 Performance por Ativo")
                if "symbol" in operations.columns and operations["symbol"].notna().any():
                    ops_with_symbol = operations[operations["symbol"].notna() & (operations["symbol"] != "")]
                    if not ops_with_symbol.empty:
                        asset_grp = ops_with_symbol.copy()
                        asset_grp["amount"] = pd.to_numeric(asset_grp["amount"], errors="coerce").fillna(0)
                        asset_grp["win"] = asset_grp["amount"] > 0
                        asset_agg = asset_grp.groupby("symbol").agg(
                            trades=("amount", "count"),
                            win_rate=("win", "mean"),
                            total_pnl=("amount", "sum"),
                        ).reset_index()
                        asset_agg["win_rate"] = (asset_agg["win_rate"] * 100).round(1)
                        asset_agg["total_pnl"] = asset_agg["total_pnl"].round(2)
                        asset_agg = asset_agg.sort_values("total_pnl", ascending=False)
                        fig_asset = px.bar(
                            asset_agg,
                            x="symbol",
                            y="total_pnl",
                            color="total_pnl",
                            color_continuous_scale=["#ef4444", "#22c55e"],
                            title="P&L total por ativo",
                            labels={"symbol": "Ativo", "total_pnl": "P&L Total (R$)"},
                        )
                        fig_asset.update_layout(template="plotly_white", coloraxis_showscale=False)
                        st.plotly_chart(fig_asset, width="stretch")
                        st.dataframe(
                            asset_agg.rename(columns={"symbol": "Ativo", "trades": "Trades", "win_rate": "Win Rate (%)", "total_pnl": "P&L Total (R$)"}),
                            width="stretch",
                            hide_index=True,
                        )
                    else:
                        st.info("Sem dados de ativo disponíveis.")
                else:
                    st.info("Sem dados de ativo disponíveis.")

            # Win rate by hour of day
            with acol2:
                st.subheader("⏰ Performance por Horário")
                df_time = operations.copy()
                df_time["amount"] = pd.to_numeric(df_time["amount"], errors="coerce").fillna(0)
                df_time["event_dt"] = pd.to_datetime(df_time["op_datetime"], errors="coerce").fillna(
                    pd.to_datetime(df_time["op_date"], errors="coerce")
                )
                df_time = df_time.dropna(subset=["event_dt"])
                if not df_time.empty:
                    df_time["hour"] = df_time["event_dt"].dt.hour
                    hour_agg = df_time.groupby("hour").agg(
                        trades=("amount", "count"),
                        pnl=("amount", "sum"),
                    ).reset_index()
                    hour_agg["resultado"] = hour_agg["pnl"].apply(lambda x: "Lucro" if x >= 0 else "Perda")
                    fig_hour = px.bar(
                        hour_agg,
                        x="hour",
                        y="pnl",
                        color="resultado",
                        color_discrete_map={"Lucro": "#22c55e", "Perda": "#ef4444"},
                        title="P&L por hora do dia",
                        labels={"hour": "Hora", "pnl": "P&L (R$)"},
                    )
                    fig_hour.update_layout(template="plotly_white")
                    st.plotly_chart(fig_hour, width="stretch")
                else:
                    st.info("Sem dados de horário disponíveis.")

            st.divider()

            # Heatmap: day of week × hour
            st.subheader("🗓️ Mapa de Calor — Dia da Semana × Horário")
            df_heat = operations.copy()
            df_heat["amount"] = pd.to_numeric(df_heat["amount"], errors="coerce").fillna(0)
            df_heat["event_dt"] = pd.to_datetime(df_heat["op_datetime"], errors="coerce").fillna(
                pd.to_datetime(df_heat["op_date"], errors="coerce")
            )
            df_heat = df_heat.dropna(subset=["event_dt"])
            if not df_heat.empty:
                day_names = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
                df_heat["weekday"] = df_heat["event_dt"].dt.weekday
                df_heat["hour"] = df_heat["event_dt"].dt.hour
                heat_pivot = df_heat.pivot_table(index="weekday", columns="hour", values="amount", aggfunc="sum", fill_value=0)
                heat_pivot.index = [day_names[i] for i in heat_pivot.index]
                fig_heat = go.Figure(
                    data=go.Heatmap(
                        z=heat_pivot.values,
                        x=[f"{h:02d}h" for h in heat_pivot.columns],
                        y=heat_pivot.index.tolist(),
                        colorscale=[[0, "#ef4444"], [0.5, "#f8fafc"], [1, "#22c55e"]],
                        zmid=0,
                        hoverongaps=False,
                        colorbar={"title": "P&L (R$)"},
                    )
                )
                fig_heat.update_layout(
                    title="P&L acumulado por dia da semana e hora",
                    template="plotly_white",
                    xaxis_title="Hora do dia",
                    yaxis_title="Dia da semana",
                )
                st.plotly_chart(fig_heat, width="stretch")
            else:
                st.info("Dados de horário insuficientes para o mapa de calor.")

            st.divider()

            # Win/loss streak table
            st.subheader("🔢 Estatísticas de Sequência")
            seq1, seq2, seq3, seq4 = st.columns(4)
            seq1.metric("Vitórias consecutivas (máx.)", stats.get("max_consec_win", 0))
            seq2.metric("Derrotas consecutivas (máx.)", stats.get("max_consec_loss", 0))
            seq3.metric("Total de vitórias", stats.get("win_count", 0))
            seq4.metric("Total de derrotas", stats.get("loss_count", 0))

    # ================================================================
    # TAB 4 — GESTÃO DE RISCO
    # ================================================================
    with tab_risk:
        st.header("🛡️ Gestão de Risco Inteligente")

        stats_risk = compute_trade_stats(operations) if not operations.empty else {}

        # Kelly Criterion
        st.subheader("📐 Critério de Kelly — Tamanho de Posição")
        if stats_risk:
            kelly_val = kelly_criterion(
                stats_risk.get("win_rate", 0),
                stats_risk.get("avg_win", 0),
                abs(stats_risk.get("avg_loss", 0)),
            )
            half_kelly = kelly_val / 2
            quarter_kelly = kelly_val / 4

            kc1, kc2, kc3 = st.columns(3)
            kc1.metric("Kelly completo", f"{kelly_val*100:.1f}% da banca")
            kc2.metric("½ Kelly (recomendado)", f"{half_kelly*100:.1f}% da banca = {format_currency(current_balance * half_kelly)}")
            kc3.metric("¼ Kelly (conservador)", f"{quarter_kelly*100:.1f}% da banca = {format_currency(current_balance * quarter_kelly)}")

            st.caption(
                "O critério de Kelly calcula o percentual ótimo de banca a arriscar por operação para maximizar o crescimento a longo prazo. "
                "Use ½ Kelly ou ¼ Kelly para reduzir volatilidade."
            )
        else:
            st.info("Importe trades para calcular o tamanho de posição pelo critério de Kelly.")

        st.divider()

        # Drawdown chart
        st.subheader("📉 Drawdown da Banca")
        if not operations.empty:
            dd_df = compute_drawdown(operations, float(settings["initial_bankroll"]))
            if not dd_df.empty:
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=dd_df["datetime"], y=dd_df["balance"],
                    name="Saldo", line={"color": "#2563eb", "width": 2},
                ))
                fig_dd.add_trace(go.Scatter(
                    x=dd_df["datetime"], y=dd_df["peak"],
                    name="Pico histórico", line={"color": "#22c55e", "width": 1, "dash": "dash"},
                ))
                fig_dd.update_layout(
                    template="plotly_white", title="Evolução do saldo vs. pico histórico",
                    yaxis_title="Saldo (R$)", xaxis_title="Data",
                )
                st.plotly_chart(fig_dd, width="stretch")

                fig_dd2 = go.Figure()
                fig_dd2.add_trace(go.Scatter(
                    x=dd_df["datetime"], y=dd_df["drawdown_pct"],
                    fill="tozeroy", name="Drawdown %",
                    line={"color": "#ef4444"},
                    fillcolor="rgba(239,68,68,0.2)",
                ))
                fig_dd2.update_layout(
                    template="plotly_white", title="Drawdown (%)",
                    yaxis_title="Drawdown (%)", xaxis_title="Data",
                )
                st.plotly_chart(fig_dd2, width="stretch")

                max_dd = float(dd_df["drawdown_pct"].min())
                current_dd = float(dd_df["drawdown_pct"].iloc[-1]) if not dd_df.empty else 0.0
                dd1, dd2 = st.columns(2)
                dd1.metric("Drawdown máximo", f"{max_dd:.2f}%")
                dd2.metric("Drawdown atual", f"{current_dd:.2f}%")
        else:
            st.info("Importe trades para visualizar o drawdown.")

        st.divider()

        # Risk recommendations
        st.subheader("💡 Recomendações de Risco")
        recs = risk_recommendations(stats_risk, settings, current_balance)
        for rec in recs:
            _render_alert(rec)

        st.divider()

        # Daily stop tracker
        st.subheader("🚦 Monitor de Stop Diário")
        risk_m1, risk_m2, risk_m3 = st.columns(3)
        risk_m1.metric("Resultado hoje", format_currency(daily_result_today), delta=f"{daily_profit_pct:+.2f}%")
        risk_m2.metric("Stop loss diário", format_currency(day_loss_limit_value), delta=f"-{settings['daily_loss_limit']:.1f}%")
        risk_m3.metric("Meta diária", format_currency(day_profit_goal_value), delta=f"+{settings['daily_profit_goal']:.1f}%")

        if stop_loss_hit_today:
            st.error("⛔ Stop loss diário atingido! Recomenda-se encerrar as operações pelo restante do dia.")
        elif goal_hit_today:
            st.success("🎯 Meta diária atingida! Considere encerrar as operações para proteger o lucro.")
        else:
            remaining_loss = day_loss_limit_value + daily_result_today
            remaining_gain = day_profit_goal_value - daily_result_today
            if day_loss_limit_value > 0:
                st.info(f"Margem restante até o stop loss: **{format_currency(max(0.0, remaining_loss))}**")
            if day_profit_goal_value > 0:
                st.info(f"Falta **{format_currency(max(0.0, remaining_gain))}** para atingir a meta diária.")

    # ================================================================
    # TAB 5 — IA & SUGESTÕES
    # ================================================================
    with tab_ai:
        st.header("🤖 Sugestões Automáticas com IA")
        st.markdown(
            "O sistema analisa seu histórico de trades e gera sugestões personalizadas "
            "baseadas em padrões estatísticos de desempenho."
        )

        stats_ai = compute_trade_stats(operations) if not operations.empty else {}
        suggestions = ai_suggestions(operations, stats_ai)

        for sug in suggestions:
            _render_alert(sug)

        if not operations.empty and stats_ai:
            st.divider()
            st.subheader("📊 Resumo Estatístico Completo")

            ai1, ai2, ai3 = st.columns(3)
            with ai1:
                st.markdown("**Desempenho geral**")
                st.write(f"- Total de trades: **{stats_ai.get('total', 0)}**")
                st.write(f"- Win rate: **{stats_ai.get('win_rate', 0):.1f}%**")
                st.write(f"- Profit factor: **{_format_profit_factor(stats_ai.get('profit_factor', 0))}**")
            with ai2:
                st.markdown("**Médias**")
                st.write(f"- Ganho médio: **{format_currency(stats_ai.get('avg_win', 0))}**")
                st.write(f"- Perda média: **{format_currency(abs(stats_ai.get('avg_loss', 0)))}**")
                st.write(f"- Ganho/Perda ratio: **{_gain_loss_ratio(stats_ai.get('avg_win', 0), stats_ai.get('avg_loss', 0))}**")
            with ai3:
                st.markdown("**Sequências**")
                st.write(f"- Máx. vitórias seguidas: **{stats_ai.get('max_consec_win', 0)}**")
                st.write(f"- Máx. derrotas seguidas: **{stats_ai.get('max_consec_loss', 0)}**")
                st.write(f"- Lucro bruto: **{format_currency(stats_ai.get('gross_profit', 0))}**")
                st.write(f"- Perda bruta: **{format_currency(stats_ai.get('gross_loss', 0))}**")

            # Best/worst asset chart
            if "symbol" in operations.columns and operations["symbol"].notna().any():
                st.divider()
                st.subheader("🏅 Ranking de Ativos")
                ops_sym = operations[operations["symbol"].notna() & (operations["symbol"] != "")].copy()
                ops_sym["amount"] = pd.to_numeric(ops_sym["amount"], errors="coerce").fillna(0)
                if not ops_sym.empty:
                    rank_df = ops_sym.groupby("symbol").agg(
                        pnl=("amount", "sum"),
                        trades=("amount", "count"),
                    ).reset_index().sort_values("pnl", ascending=True)
                    fig_rank = px.bar(
                        rank_df,
                        x="pnl",
                        y="symbol",
                        orientation="h",
                        color="pnl",
                        color_continuous_scale=["#ef4444", "#22c55e"],
                        title="P&L total por ativo (do pior ao melhor)",
                        labels={"pnl": "P&L Total (R$)", "symbol": "Ativo"},
                    )
                    fig_rank.update_layout(template="plotly_white", coloraxis_showscale=False)
                    st.plotly_chart(fig_rank, width="stretch")


if __name__ == "__main__":
    main()
