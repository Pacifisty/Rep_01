import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
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
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        _ensure_column(conn, "operations", "op_datetime", "op_datetime TEXT")
        _ensure_column(conn, "operations", "source", "source TEXT NOT NULL DEFAULT 'manual'")
        _ensure_column(conn, "operations", "external_id", "external_id TEXT")
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
) -> bool:
    with get_connection() as conn:
        before = conn.total_changes
        conn.execute(
            """
            INSERT OR IGNORE INTO operations (op_date, op_datetime, description, amount, source, external_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                op_date.isoformat(),
                op_datetime.isoformat(timespec="seconds") if op_datetime else None,
                description.strip(),
                amount,
                source,
                external_id,
            ),
        )
        return conn.total_changes > before


def remove_operation(op_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM operations WHERE id = ?", (op_id,))


def load_operations() -> pd.DataFrame:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, op_date, op_datetime, description, amount, source, external_id, created_at FROM operations ORDER BY op_date, id"
        ).fetchall()

    if not rows:
        return pd.DataFrame(columns=["id", "op_date", "op_datetime", "description", "amount", "source", "external_id", "created_at"])

    return pd.DataFrame([dict(r) for r in rows])


# ---------- IQ Option Sync ----------
def _extract_operations_from_payload(payload: Any, default_source: str = "iqoption") -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for child in node:
                walk(child)
            return

        if not isinstance(node, dict):
            return

        profit = None
        for key in ("close_profit", "profit_amount", "profit", "pnl", "win_amount"):
            if key in node and isinstance(node[key], (int, float)):
                profit = float(node[key])
                break

        timestamp = None
        for key in ("close_time", "close_time_ms", "closeTimestamp", "timestamp", "created", "created_at"):
            if key in node:
                value = node[key]
                if isinstance(value, (int, float)):
                    timestamp = float(value)
                elif isinstance(value, str):
                    try:
                        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
                    except ValueError:
                        timestamp = None
                break

        if profit is not None and timestamp is not None:
            if timestamp > 10_000_000_000:
                timestamp = timestamp / 1000
            op_date = datetime.utcfromtimestamp(timestamp).date()
            external_id = str(node.get("id") or node.get("position_id") or node.get("order_id") or f"{int(timestamp)}_{profit}")
            symbol = node.get("active") or node.get("instrument") or node.get("symbol") or "ATIVO"
            operation_type = node.get("type") or node.get("instrument_type") or "trade"
            desc = f"IQ Option | {operation_type} | {symbol}"
            items.append(
                {
                    "op_date": op_date,
                    "op_datetime": datetime.utcfromtimestamp(timestamp),
                    "description": desc,
                    "amount": profit,
                    "source": default_source,
                    "external_id": external_id,
                }
            )

        for value in node.values():
            walk(value)

    walk(payload)
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        unique[(item["source"], item["external_id"])] = item
    return list(unique.values())


def fetch_iqoption_operations(email: str, password: str, limit: int = 100, balance_mode: str = "REAL") -> list[dict[str, Any]]:
    try:
        from iqoptionapi.stable_api import IQ_Option
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Biblioteca iqoptionapi não instalada. Rode: pip install -r requirements.txt") from exc

    api = IQ_Option(email, password)
    connected, reason = api.connect()
    if not connected:
        raise RuntimeError(f"Falha ao conectar na IQ Option: {reason}")
    api.change_balance(balance_mode)  # "REAL" ou "PRACTICE"

    payloads: list[Any] = []

    methods_to_try = [
        ("get_position_history_v2", ("binary-option", limit, 0)),
        ("get_position_history", ("binary-option", limit, 0, 0)),
        ("get_position_history_v2", ("digital-option", limit, 0)),
        ("get_position_history", ("digital-option", limit, 0, 0)),
        ("get_position_history_v2", ("forex", limit, 0)),
        ("get_position_history_v2", ("crypto", limit, 0)),
    ]

    for method_name, args in methods_to_try:
        method = getattr(api, method_name, None)
        if callable(method):
            try:
                result = method(*args)
                print("METODO:", method_name)
                print("TIPO:", type(result))
                print("RESULTADO:", result)
                if result:
                    payloads.append(result)
            except Exception as e:
                print("ERRO NO MÉTODO", method_name, e)
                continue

    if hasattr(api, "close_connect"):
        api.close_connect()

    operations: list[dict[str, Any]] = []
    for payload in payloads:
        extracted = _extract_operations_from_payload(payload)
        print("EXTRAIDAS:", len(extracted))
        operations.extend(extracted)

    unique_by_key = {(op["source"], op["external_id"]): op for op in operations}
    return list(unique_by_key.values())


# ---------- UI ----------
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
    st.set_page_config(page_title="Controle de Banca", page_icon="💰", layout="wide")
    init_db()

    st.title("💰 Controle de Banca - Ganhos e Perdas")
    st.caption(f"Sistema local para gerenciamento diário da sua banca. Modo atual: {VTEST_LABEL}.")

    settings = load_settings()
    operations = load_operations()
    if "hide_values" not in st.session_state:
        st.session_state.hide_values = False

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
    start_of_day_balance = float(settings["initial_bankroll"]) + net_until_yesterday

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

        st.divider()
        st.subheader("🔌 IQ Option")
        iq_email = st.text_input("Email IQ Option", key="iq_email")
        iq_password = st.text_input("Senha IQ Option", type="password", key="iq_password")
        iq_balance_mode = st.selectbox("Conta para sincronização", options=["REAL", "PRACTICE"], index=0)
        iq_limit = st.number_input("Máx. operações para buscar", min_value=10, max_value=500, value=500, step=10)
        if st.button("Sincronizar operações da IQ Option", use_container_width=True):
            if not iq_email or not iq_password:
                st.error("Preencha email e senha da IQ Option para sincronizar.")
            else:
                with st.spinner("Sincronizando operações da IQ Option..."):
                    try:
                        iq_ops = fetch_iqoption_operations(iq_email, iq_password, limit=int(iq_limit), balance_mode=iq_balance_mode)
                        st.write("Total retornado pela API:", len(iq_ops))
                        st.write(iq_ops[:5] if iq_ops else "Nenhuma operação extraída")
                        inserted = 0
                        for op in iq_ops:
                            did_insert = add_operation(
                                op["op_date"],
                                op["description"],
                                float(op["amount"]),
                                op["source"],
                                op["external_id"],
                                op.get("op_datetime"),
                            )
                            if did_insert:
                                inserted += 1
                        st.success(f"Sincronização concluída. {inserted} novas operações importadas.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Erro na sincronização com IQ Option: {exc}")

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

    st.subheader("Registrar operação")
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

    st.subheader("Histórico de operações")
    if operations.empty:
        st.info("Nenhuma operação registrada ainda.")
    else:
        display_df = operations.copy()
        display_df["Data/Hora"] = pd.to_datetime(display_df["op_datetime"], errors="coerce").dt.strftime("%d/%m/%Y %H:%M")
        display_df["Data/Hora"] = display_df["Data/Hora"].fillna(pd.to_datetime(display_df["op_date"]).dt.strftime("%d/%m/%Y"))
        display_df["Valor"] = display_df["amount"].apply(format_currency)
        display_df = display_df.rename(columns={"id": "ID", "description": "Descrição", "source": "Origem"})
        st.dataframe(display_df[["ID", "Data/Hora", "Descrição", "Origem", "Valor"]], use_container_width=True, hide_index=True)

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

        st.subheader("📈 Desempenho detalhado das operações")
        timeframe = st.selectbox("Visualizar por", ["Hora", "Dia", "Semana", "Mês", "Ano"], index=1)
        fig, grouped = build_detailed_chart(operations, float(settings["initial_bankroll"]), timeframe)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            grouped.rename(
                columns={
                    "event_dt": "Período",
                    "pnl_periodo": "P&L do período",
                    "saldo_acumulado": "Saldo acumulado",
                    "resultado": "Resultado",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Resumo líquido")
    st.write(f"**Saldo líquido:** {maybe_mask_currency(net_balance, hide_values)}")

    selected_date = st.date_input("Verificar alertas do dia", value=date.today(), key="alert_date")
    daily_result = 0.0
    if not operations.empty:
        day_ops = operations[operations["op_date"] == selected_date.isoformat()]
        daily_result = float(day_ops["amount"].sum()) if not day_ops.empty else 0.0

    st.write(f"Resultado do dia {selected_date.strftime('%d/%m/%Y')}: **{maybe_mask_currency(daily_result, hide_values)}**")

    loss_limit_value = day_loss_limit_value
    profit_goal_value = day_profit_goal_value

    st.caption(
        f"Limites do dia (base no saldo atual): stop loss = {format_currency(loss_limit_value)} "
        f"({settings['daily_loss_limit']:.2f}%), meta = {format_currency(profit_goal_value)} "
        f"({settings['daily_profit_goal']:.2f}%)."
    )

    if settings["daily_loss_limit"] > 0 and daily_result <= -loss_limit_value:
        st.error("⚠️ Stop loss diário atingido!")

    if settings["daily_profit_goal"] > 0 and daily_result >= profit_goal_value:
        st.success("🎯 Meta de lucro diária atingida!")


if __name__ == "__main__":
    main()
