import pandas as pd
import streamlit as st
from pathlib import Path
import json
import sqlite3
import numpy as np
import math

try:
    import chromadb
except Exception:
    chromadb = None

try:
    import altair as alt
except Exception:
    alt = None


def symptom_to_score(series: pd.Series) -> pd.Series:
    """Convert mixed symptom text/numeric values into numeric severity."""
    mapping = {
        "not at all": 0,
        "very low/little": 1,
        "low": 2,
        "moderate": 3,
        "high": 4,
        "very high": 5,
    }
    lower = series.astype(str).str.strip().str.lower()
    mapped = lower.map(mapping)
    numeric = pd.to_numeric(series, errors="coerce")
    return mapped.fillna(numeric)


st.set_page_config(page_title="Innovation Dashboard MVP", layout="wide")
st.title("Innovation Dashboard (MVP)")
st.caption(
    "Local ready+kpi as primary data source, with Stage2 auto-detect extension")


@st.cache_data
def load_csv_data(filename: str) -> pd.DataFrame:
    """Load CSV from project root or data_processed folder."""
    base = Path(__file__).parent
    candidates = [base / filename, base / "data_processed" / filename]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(
        f"{filename} not found in project root or data_processed/")


@st.cache_data
def load_kpi_data() -> pd.DataFrame:
    return load_csv_data("dashboard_kpi.csv")


@st.cache_data
def load_ready_data() -> pd.DataFrame:
    return load_csv_data("dashboard_ready.csv")


def load_stage2_profile(base: Path, user_id: int):
    """Try to find a Stage2 profile JSON for the selected user."""
    stage2_dir = base / "stage2_users"
    if not stage2_dir.exists():
        return None, "stage2_users directory missing"

    exact_candidates = [
        stage2_dir / f"{user_id}.json",
        stage2_dir / f"user_{user_id}.json",
        stage2_dir / f"id_{user_id}.json",
    ]
    for p in exact_candidates:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8")), str(p.name)
            except Exception:
                return None, f"failed to parse {p.name}"

    # fallback: scan json files and match id field
    for p in stage2_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(data.get("id")) == str(user_id):
            return data, str(p.name)

    return None, "no matching Stage2 json"


def _find_id_column(df: pd.DataFrame) -> str:
    """Find id column in various naming formats."""
    for col in ["id", "user_id", "patient_id", "ID"]:
        if col in df.columns:
            return col
    return None


def _get_user_record(df: pd.DataFrame, user_id: int) -> tuple:
    """Extract user record from dataframe."""
    id_col = _find_id_column(df)
    if not id_col:
        return None, "id column not found"
    user_data = df[df[id_col] == user_id]
    if user_data.empty:
        return None, f"user {user_id} not found"
    return user_data.iloc[0].to_dict(), "loaded"


def load_user_data_from_csv(base: Path, filename: str, user_id: int) -> tuple:
    """Load user data from CSV file (searches root and data_processed)."""
    csv_files = [base / filename, base / "data_processed" / filename]
    for csv_file in csv_files:
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
                return _get_user_record(df, user_id)
            except Exception as e:
                return None, f"error reading {csv_file.name}: {e}"
    return None, f"{filename} not found"


def load_user_data_from_json(base: Path, filenames: list, user_id: int) -> tuple:
    """Load user data from JSON file (supports single object or array)."""
    for filename in filenames:
        json_files = [base / filename, base / "data_processed" / filename]
        for json_file in json_files:
            if json_file.exists():
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        if str(data.get("id")) == str(user_id):
                            return data, "loaded from JSON"
                        return None, f"user {user_id} not in JSON"
                    elif isinstance(data, list):
                        for item in data:
                            if str(item.get("id")) == str(user_id):
                                return item, "loaded from JSON"
                        return None, f"user {user_id} not in JSON array"
                except Exception as e:
                    return None, f"error reading {json_file.name}: {e}"
    return None, "JSON file not found"


def load_prediction_data(base: Path, user_id: int):
    """Load prediction data from prediction_output.csv."""
    data, hint = load_user_data_from_csv(
        base, "prediction_output.csv", user_id)
    return data, hint


def load_triage_data(base: Path, user_id: int):
    """Load triage data from triage_output.csv/json."""
    # Try CSV first
    data, hint = load_user_data_from_csv(base, "triage_output.csv", user_id)
    if data is not None:
        return data, "loaded from CSV"
    # Try JSON as fallback
    data, hint = load_user_data_from_json(
        base, ["triage_output.json"], user_id)
    return data, hint


def fmt_num(value, digits=2):
    if pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}"


def fmt_completeness_pct(value) -> str:
    """Normalize completeness to percent string for clearer KPI display."""
    if pd.isna(value):
        return "N/A"
    v = float(value)

    if 0 <= v <= 1:
        pct = v * 100
    elif 1 < v <= 120 and float(v).is_integer():
        # Treat small integer values as recorded-day count in a 60-day window.
        pct = min(v, 60.0) / 60.0 * 100.0
    elif 0 <= v <= 100:
        pct = v
    else:
        return fmt_num(v, 0)

    return f"{pct:.0f}%"


def _extract_field(data: dict, possible_names: list):
    """Extract field from dict using possible column names."""
    if not data:
        return None
    for col in possible_names:
        if col in data:
            return data[col]
    return None
    """Map symptom text levels to numeric values for trend plotting."""
    mapping = {
        "not at all": 0,
        "very low/little": 1,
        "low": 2,
        "moderate": 3,
        "high": 4,
        "very high": 5,
    }
    lower = series.astype(str).str.strip().str.lower()
    mapped = lower.map(mapping)
    numeric = pd.to_numeric(series, errors="coerce")
    return mapped.fillna(numeric)


def keyword_context_snippet(full_text: str, keyword: str, radius: int = 200) -> str:
    """Build a short preview centered around the first keyword hit."""
    if not full_text:
        return ""

    text = full_text.replace("\n", " ").strip()
    if not keyword.strip():
        return (text[:240] + "...") if len(text) > 240 else text

    lower_text = text.lower()
    lower_key = keyword.strip().lower()
    idx = lower_text.find(lower_key)

    if idx == -1:
        return (text[:240] + "...") if len(text) > 240 else text

    start = max(0, idx - radius)
    end = min(len(text), idx + len(lower_key) + radius)
    snippet = text[start:end]

    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."

    return snippet


def search_stage3_knowledge(base: Path, keyword: str, k: int = 3):
    """Simple keyword search over local chroma_db files (no LLM required)."""
    chroma_dir = base / "chroma_db"
    if not chroma_dir.exists():
        return []

    key = keyword.strip().lower()
    if not key:
        return []

    matches = []
    for p in chroma_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".txt", ".md", ".json", ".csv", ".yaml", ".yml"}:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        idx = text.lower().find(key)
        if idx == -1:
            continue
        start = max(0, idx - 60)
        end = min(len(text), idx + 120)
        full_text = text.replace("\n", " ").strip()
        snippet = keyword_context_snippet(full_text, keyword, radius=200)
        matches.append({
            "title": p.name,
            "source": str(p.relative_to(base)),
            "snippet": snippet,
            "full_text": full_text,
        })

    return matches[:k]


def find_chroma_db_dir(base: Path):
    """Find chroma db directory by checking common locations."""
    candidates = [
        base / "chroma_db",
        Path.cwd() / "chroma_db",
    ]
    for c in candidates:
        if (c / "chroma.sqlite3").exists():
            return c
    return None


def get_chroma_collections(chroma_dir: Path):
    """Read collection names directly from chroma sqlite."""
    db_path = chroma_dir / "chroma.sqlite3"
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='collections'")
        if cur.fetchone() is None:
            return []
        cur.execute("SELECT name FROM collections ORDER BY name")
        return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


def search_chroma_sqlite(chroma_dir: Path, keyword: str, k: int = 3):
    """Keyword search snippets from Chroma FTS table (no LLM)."""
    db_path = chroma_dir / "chroma.sqlite3"
    if not db_path.exists() or not keyword.strip():
        return []

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_fulltext_search'")
        if cur.fetchone() is None:
            return []

        pattern = f"%{keyword.strip()}%"
        cur.execute(
            "SELECT string_value FROM embedding_fulltext_search WHERE string_value LIKE ? LIMIT ?",
            (pattern, k),
        )
        rows = cur.fetchall()
        results = []
        for i, (text,) in enumerate(rows, 1):
            if not text:
                continue
            full_text = text.replace("\n", " ").strip()
            snippet = keyword_context_snippet(full_text, keyword, radius=200)
            results.append({
                "title": f"Hit {i}",
                "source": str(db_path.relative_to(base_dir)),
                "snippet": snippet,
                "full_text": full_text,
            })
        return results
    finally:
        conn.close()


def search_chroma_vector(chroma_dir: Path, keyword: str, k: int = 3, collection_name: str = "medical_knowledge"):
    """True vector retrieval using Chroma query_embeddings.

    Strategy (no DB rebuild):
    1) Use where_document contains keyword to fetch seed chunks + stored embeddings.
    2) Compute centroid embedding from seed chunks.
    3) Run query_embeddings to retrieve top-k nearest chunks.
    """
    if chromadb is None or not keyword.strip():
        return [], "chromadb package unavailable"

    db_path = chroma_dir / "chroma.sqlite3"
    if not db_path.exists():
        return [], "chroma sqlite missing"

    try:
        client = chromadb.PersistentClient(path=str(chroma_dir))
        col_names = [c.name for c in client.list_collections()]
        target_name = collection_name if collection_name in col_names else (
            col_names[0] if col_names else None)
        if target_name is None:
            return [], "no collection found"

        collection = client.get_collection(target_name)

        # Build seed set robustly: query full phrase first, then fallback to individual tokens.
        token_candidates = [keyword.strip()] + \
            [t for t in keyword.strip().split() if t]
        seed_emb_list = []
        for token in token_candidates:
            seed = collection.get(
                where_document={"$contains": token},
                include=["embeddings", "documents", "metadatas"],
                limit=max(10, k * 5),
            )
            embs = seed.get("embeddings")
            if embs is not None and len(embs) > 0:
                seed_emb_list.extend(list(np.asarray(embs, dtype=float)))
            if len(seed_emb_list) >= max(10, k * 5):
                break

        emb_count = len(seed_emb_list)
        if emb_count == 0:
            return [], "no seed embeddings from keyword"

        centroid = np.asarray(seed_emb_list, dtype=float).mean(axis=0).tolist()
        out = collection.query(
            query_embeddings=[centroid],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        docs = out.get("documents", [[]])[0]
        metas = out.get("metadatas", [[]])[0]
        dists = out.get("distances", [[]])[0]

        results = []
        for i, full_text in enumerate(docs, 1):
            if not full_text:
                continue
            md = metas[i - 1] if i - 1 < len(metas) and metas[i - 1] else {}
            source = md.get("source") or md.get(
                "url") or f"collection:{target_name}"
            title = md.get("title") or md.get("section") or f"Hit {i}"
            snippet = keyword_context_snippet(full_text, keyword, radius=200)
            distance = dists[i - 1] if i - 1 < len(dists) else None

            results.append({
                "title": title,
                "source": str(source),
                "snippet": snippet,
                "full_text": full_text,
                "distance": distance,
            })

        return results, f"vector search on collection `{target_name}`"
    except Exception as exc:
        return [], f"vector search error: {exc}"


def retrieve_stage3_hits(base: Path, query: str, k: int = 3):
    """Unified retrieval helper with vector-first fallback chain."""
    chroma_dir_local = find_chroma_db_dir(base)
    if not query.strip():
        return [], "empty query", chroma_dir_local

    if chroma_dir_local is not None:
        hits, mode = search_chroma_vector(chroma_dir_local, query, k=k)
        if hits:
            return hits, mode, chroma_dir_local

        hits = search_chroma_sqlite(chroma_dir_local, query, k=k)
        if hits:
            return hits, "sqlite keyword fallback", chroma_dir_local

    hits = search_stage3_knowledge(base, query, k=k)
    if hits:
        return hits, "file keyword fallback", chroma_dir_local

    return [], "no hits", chroma_dir_local


def select_best_evidence_hit(hits, query: str):
    """Pick best evidence: prefer keyword-containing hit, then smaller distance."""
    if not hits:
        return None

    q = query.strip().lower()
    q_tokens = [t for t in q.split() if t]

    def has_keyword(item):
        text = f"{item.get('snippet', '')} {item.get('full_text', '')}".lower()
        if q and q in text:
            return True
        return any(tok in text for tok in q_tokens)

    scored = []
    for i, item in enumerate(hits):
        contains = has_keyword(item)
        dist = item.get("distance")
        dist_val = float(dist) if dist is not None else 9999.0
        scored.append((0 if contains else 1, dist_val, i, item))

    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    return scored[0][3]


def render_trend_chart(title: str, data: pd.DataFrame, y_col: str, window_days: int, tick_step: int):
    """Render trend chart with explicit x-range and fixed ticks on relative day axis."""
    st.markdown(f"**{title}**")
    if alt is None:
        st.line_chart(data.set_index("relative_day")[[y_col]], height=220)
        return

    x_min = -(window_days - 1)
    x_max = 0
    ticks = list(range(x_min, x_max + 1, tick_step))
    if 0 not in ticks:
        ticks.append(0)
    ticks = sorted(set(ticks))

    chart = (
        alt.Chart(data)
        .mark_line(point=False)
        .encode(
            x=alt.X(
                "relative_day:Q",
                title="relative_day (0 = latest)",
                scale=alt.Scale(domain=[x_min, x_max]),
                axis=alt.Axis(values=ticks),
            ),
            y=alt.Y(f"{y_col}:Q", title=y_col),
        )
        .properties(height=220)
    )
    st.altair_chart(chart, use_container_width=True)


try:
    df = load_kpi_data()
except Exception as exc:
    st.error(f"数据加载失败: {exc}")
    st.info("请先运行: `python3 step1_features.py` 生成 `dashboard_kpi.csv`")
    st.stop()

if "id" not in df.columns:
    st.error("`dashboard_kpi.csv` 缺少 `id` 列")
    st.stop()

try:
    ready_df = load_ready_data()
except Exception:
    ready_df = None

base_dir = Path(__file__).parent

st.success(f"Loaded KPI table: {df.shape[0]} users x {df.shape[1]} columns")
st.sidebar.header("Data Source")
st.sidebar.write("Primary: local pipeline")
st.sidebar.code("data_processed/dashboard_kpi.csv")

if ready_df is not None:
    st.sidebar.write("Timeseries: available")
else:
    st.sidebar.write("Timeseries: missing dashboard_ready.csv")

if (base_dir / "stage2_users").exists():
    stage2_count = len(list((base_dir / "stage2_users").glob("*.json")))
    st.sidebar.write(f"Stage2 profiles detected: {stage2_count}")
else:
    st.sidebar.write("Stage2 profiles: pending")

st.sidebar.header("User Selector")
user_ids = sorted(df["id"].dropna().astype(int).unique())
selected_user = st.sidebar.selectbox("Choose User ID", user_ids)

row = df[df["id"].astype(int) == int(selected_user)]
if row.empty:
    st.warning("Selected user not found in KPI table.")
    st.stop()

u = row.iloc[0]

stage2_profile, stage2_hint = load_stage2_profile(base_dir, int(selected_user))
triage_data, triage_hint = load_triage_data(base_dir, int(selected_user))

st.subheader("Triage / Risk Assessment")
if triage_data is not None:
    st.success("Status: ready")
    triage_level = _extract_field(
        triage_data, ["level", "triage_level", "severity"])
    triage_reasons = _extract_field(
        triage_data, ["reasons", "reason", "explanation"])
    if triage_level is not None:
        st.write(f"**Risk Level**: {triage_level}")
    if triage_reasons is not None:
        st.write(f"**Risk Reasons**: {triage_reasons}")

    with st.expander("Full triage data"):
        st.json(triage_data)
else:
    st.info("Status: pending")
    st.caption(f"Waiting for triage data ({triage_hint})")

st.markdown("---")
st.subheader("Stage2 Profile (Optional Extension)")
if stage2_profile is None:
    st.info(f"Stage2 user profile pending ({stage2_hint}).")
else:
    chief = stage2_profile.get("chief_concern", "N/A")
    tried = stage2_profile.get("what_tried", "N/A")
    summary = stage2_profile.get("summary", "N/A")
    st.success(f"Loaded Stage2 profile: {stage2_hint}")
    st.write(f"Chief concern: {chief}")
    st.write(f"What tried: {tried}")
    st.write(f"Summary: {summary}")

st.markdown("---")
st.subheader(f"User {int(selected_user)} - KPI Cards")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Completeness (60d)", fmt_completeness_pct(
        u.get("data_completeness_60d")))
    st.caption("% of days with valid records in last 60 days")
    st.metric("Cramp Mean (30d)", fmt_num(u.get("cramps_mean_30d"), 2))
    st.metric("Cramp Max (30d)", fmt_num(u.get("cramps_max_30d"), 2))

with col2:
    st.metric("Mood Mean (30d)", fmt_num(u.get("moodswing_mean_30d"), 2))
    st.metric("Mood Std (30d)", fmt_num(u.get("moodswing_std_30d"), 2))
    st.metric("Heavy Flow Days (30d)", fmt_num(
        u.get("flow_heavy_count_30d"), 0))

with col3:
    st.metric("Sleep Score (14d)", fmt_num(u.get("sleep_score_mean_14d"), 2))
    st.metric("Steps (7d avg)", fmt_num(u.get("steps_mean_7d"), 0))
    st.metric("Active Minutes (7d avg)", fmt_num(
        u.get("active_min_mean_7d"), 2))

with col4:
    st.metric("Resting HR (14d)", fmt_num(u.get("rhr_mean_14d"), 2))
    st.metric("Stress (14d)", fmt_num(u.get("stress_mean_14d"), 2))
    st.metric("Latest Flow Day", fmt_num(u.get("latest_flow_day"), 0))

st.markdown("---")
st.subheader("Quick Checks")

alerts = []
if pd.notna(u.get("data_completeness_60d")) and u.get("data_completeness_60d") < 40:
    alerts.append("Data completeness in last 60 days is low (< 40 days).")

if pd.notna(u.get("sleep_score_delta_7d")) and u.get("sleep_score_delta_7d") <= -8:
    alerts.append("Sleep score delta_7d <= -8 (warning threshold).")
elif pd.notna(u.get("sleep_score_delta_7d")) and u.get("sleep_score_delta_7d") < -5:
    alerts.append("Sleep score dropped significantly in recent 7 days.")

if pd.notna(u.get("steps_delta_7d")) and u.get("steps_delta_7d") < -2000:
    alerts.append("Steps trend is down (recent 7d vs previous 7d).")
elif pd.notna(u.get("steps_delta_7d")) and u.get("steps_delta_7d") < 0:
    alerts.append("Steps trend is mildly down (7d delta < 0).")

# Interpret cramp scale adaptively: use >=8 for 0-10 style scale, >=4 for 0-5 style scale.
cramp_max = u.get("cramps_max_30d")
if pd.notna(cramp_max):
    cramp_threshold = 8 if float(cramp_max) > 5 else 4
    if float(cramp_max) >= cramp_threshold:
        alerts.append(f"Cramps max (30d) is high (>= {cramp_threshold}).")

if pd.notna(u.get("stress_mean_14d")) and u.get("stress_mean_14d") > 70:
    alerts.append("Stress level appears elevated in last 14 days.")

if pd.notna(u.get("sleep_score_delta_7d")) and u.get("sleep_score_delta_7d") < 0:
    alerts.append("Sleep score trend is down (7d delta < 0).")

if ready_df is not None and "resting_heart_rate" in ready_df.columns:
    baseline_df = ready_df[ready_df["id"].astype(int) == int(selected_user)]
    baseline_rhr = pd.to_numeric(
        baseline_df["resting_heart_rate"], errors="coerce").mean()
    current_rhr = u.get("rhr_mean_14d")
    if pd.notna(baseline_rhr) and pd.notna(current_rhr) and float(current_rhr) >= float(baseline_rhr) + 5:
        alerts.append("Resting HR (14d) is above personal baseline (+5 bpm).")

if not alerts:
    st.success("No critical quick-check alerts for this user.")
else:
    for msg in alerts:
        st.warning(msg)

with st.expander("Show selected user raw KPI row"):
    st.dataframe(row, use_container_width=True)

st.markdown("---")
st.subheader("Trend Charts")

if ready_df is None:
    st.warning(
        "No timeseries table found. Put `dashboard_ready.csv` in project root or `data_processed/`.")
else:
    user_ts = ready_df[ready_df["id"].astype(int) == int(selected_user)].copy()
    user_ts = user_ts.sort_values("day_in_study")

    if user_ts.empty:
        st.warning("No timeseries rows found for this user.")
    else:
        max_day = int(user_ts["day_in_study"].max())
        st.caption("Monitoring panel: 3-5 time-series charts over recent days")
        day_window = st.slider(
            "Trend window (days)",
            min_value=14,
            max_value=min(120, max_day),
            value=min(60, max_day),
            step=1,
            key="trend_window_days",
        )
        tick_step = st.selectbox(
            "X tick interval (days)",
            options=[5, 10, 15, 20],
            index=1,
            key="trend_tick_step",
        )
        start_day = max_day - day_window + 1
        window_ts = user_ts[user_ts["day_in_study"] >= start_day].copy()
        window_ts["relative_day"] = window_ts["day_in_study"] - max_day
        st.caption(
            "X-axis locked to [-window+1, 0] in relative_day (0 = latest, negative = days ago)")

        chart_defs = [
            ("steps", "Steps (recent window)"),
            ("overall_score", "Sleep score (recent window)"),
            ("resting_heart_rate", "Resting HR (recent window)"),
            ("stress_score", "Stress score (recent window)"),
        ]

        plot_count = 0
        left, right = st.columns(2)
        targets = [left, right]
        target_idx = 0

        for col, title in chart_defs:
            if col not in window_ts.columns:
                continue
            plot_df = window_ts[["relative_day", col]].dropna()
            if plot_df.empty:
                continue
            with targets[target_idx % 2]:
                render_trend_chart(title, plot_df, col, day_window, tick_step)
            target_idx += 1
            plot_count += 1

        # Optional symptom trend (5th chart candidate)
        symptom_series = None
        symptom_name = None
        if "cramps" in window_ts.columns:
            symptom_series = symptom_to_score(window_ts["cramps"])
            symptom_name = "Cramps severity"
        elif "moodswing" in window_ts.columns:
            symptom_series = symptom_to_score(window_ts["moodswing"])
            symptom_name = "Mood swing severity"

        if symptom_series is not None:
            symptom_df = pd.DataFrame({
                "relative_day": window_ts["relative_day"],
                "symptom_score": symptom_series,
            }).dropna()
            if not symptom_df.empty:
                with targets[target_idx % 2]:
                    render_trend_chart(
                        f"{symptom_name} (optional)", symptom_df, "symptom_score", day_window, tick_step)
                plot_count += 1

        if plot_count == 0:
            st.warning(
                "No plottable trend columns found in dashboard_ready.csv.")

def prepare_stage4_request(user_id: int, kpi_row: dict, alerts_list: list, 
                           triage_data: dict, stage2_profile: dict, 
                           evidence_query: str) -> dict:
    """Prepare request payload for Stage 4 RAG + LLM pipeline.
    
    This packages frontend data into format expected by Stage 4's rag_api.py
    """
    request_payload = {
        "user_id": int(user_id),
        "query_mode": "doctor_summary",
        "evidence_keyword": evidence_query.strip() if evidence_query else "",
        
        # KPI snapshot (recent 14/30 days)
        "kpi_snapshot": {
            "sleep_score_mean_14d": fmt_num(kpi_row.get('sleep_score_mean_14d'), 2),
            "sleep_score_delta_7d": fmt_num(kpi_row.get('sleep_score_delta_7d'), 2),
            "steps_mean_7d": fmt_num(kpi_row.get('steps_mean_7d'), 0),
            "steps_delta_7d": fmt_num(kpi_row.get('steps_delta_7d'), 0),
            "rhr_mean_14d": fmt_num(kpi_row.get('rhr_mean_14d'), 2),
            "stress_mean_14d": fmt_num(kpi_row.get('stress_mean_14d'), 2),
            "cramps_max_30d": fmt_num(kpi_row.get('cramps_max_30d'), 2),
        },
        
        # Alerts from quick checks
        "alerts": alerts_list[:5],  # Top 5 alerts
        
        # Stage 2 profile (patient context)
        "stage2_profile": stage2_profile.copy() if isinstance(stage2_profile, dict) else None,
        
        # Risk assessment from Stage 2
        "triage_data": triage_data.copy() if isinstance(triage_data, dict) else None,
    }
    return request_payload


def call_stage4_rag_api(request_payload: dict) -> tuple:
    """Call Stage 4's rag_api.py to generate doctor summary using RAG + LLM.
    
    Returns: (summary_markdown, raw_summary_text, success)
    """
    # TODO: Replace with actual API call to Stage 4
    # For now, returning placeholder indicating waiting for Stage 4 implementation
    
    try:
        # Placeholder: would call something like:
        # response = requests.post("http://localhost:8000/api/doctor_summary", 
        #                         json=request_payload, timeout=30)
        # if response.status_code == 200:
        #     result = response.json()
        #     return result["markdown"], result["raw_text"], True
        
        st.warning("⏳ Stage 4 RAG + LLM pipeline not yet connected")
        return None, None, False
        
    except Exception as e:
        st.error(f"Error calling Stage 4 API: {e}")
        return None, None, False


st.markdown("---")
st.subheader("Summary for Doctor")

evidence_query = st.text_input(
    "Query keyword for evidence retrieval (optional)",
    value="cycle",
    key="doctor_evidence_query",
)

if st.button("Generate Summary for Doctor"):
    # Prepare request for Stage 4
    request_payload = prepare_stage4_request(
        user_id=int(selected_user),
        kpi_row=u,
        alerts_list=alerts,
        triage_data=triage_data,
        stage2_profile=stage2_profile,
        evidence_query=evidence_query
    )
    
    # Call Stage 4 RAG + LLM pipeline
    with st.spinner("🔄 Generating medical summary with RAG + LLM..."):
        summary_md, summary_raw, success = call_stage4_rag_api(request_payload)
    
    if success and summary_md is not None:
        # Display generated summary
        st.markdown(summary_md)
        
        with st.expander("Raw summary (copy/export)"):
            st.text_area("Summary Output", value=summary_raw, height=280)
        
        st.download_button(
            label="Download summary.txt",
            data=summary_raw,
            file_name=f"summary_user_{int(selected_user)}.txt",
            mime="text/plain",
        )
    else:
        st.info("💡 Waiting for Stage 4 implementation. "
               "Connect rag_api.py to enable AI-generated summaries.")

st.markdown("---")
st.subheader("📚 Relevant Medical References")

chroma_dir = find_chroma_db_dir(base_dir)
if chroma_dir is not None:
    with st.expander("🔧 Knowledge Base Status", expanded=False):
        st.success("✓ Knowledge base available")
        db_path = chroma_dir / "chroma.sqlite3"
        st.caption(f"Database: {db_path.name}")
        collections = get_chroma_collections(chroma_dir)
        if collections and "medical_knowledge" in collections:
            st.success("✓ Medical knowledge collection found")
        else:
            st.warning("Collection metadata unavailable")
    
    top_k = st.slider("Number of references", min_value=1, max_value=5, value=3, step=1, key="kb_slider")
    query = st.text_input("Search for medical information", value="cycle", placeholder="e.g., pain, fever, medication...", key="kb_query")
    
    if query.strip():
        kb_results, retrieval_mode, _ = retrieve_stage3_hits(base_dir, query, k=top_k)
        if not kb_results:
            st.info(f"ℹ️ No references found for '{query}'. Try different keywords.")
        else:
            st.markdown(f"### Found {len(kb_results)} relevant reference(s)")
            
            for i, item in enumerate(kb_results, 1):
                with st.container():
                    col1, col2 = st.columns([0.8, 0.2])
                    with col1:
                        st.markdown(f"**Reference {i}: {item['title']}**")
                        st.write(item["snippet"])
                    with col2:
                        if item.get("distance") is not None:
                            relevance = max(0, 100 - int(item.get("distance", 0) * 100))
                            st.metric("Relevance", f"{relevance}%")
                    
                    with st.expander("📖 Full text & source"):
                        st.write(item.get("full_text", item.get("snippet", "")))
                        st.caption(f"📍 Source: {item['source']}")
                    st.divider()
else:
    st.info("📚 Medical knowledge base not yet loaded (Stage3 pending). Knowledge base will be available once initialized.")
