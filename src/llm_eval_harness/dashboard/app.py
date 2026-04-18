"""Streamlit dashboard for exploring LLM eval results."""

from __future__ import annotations

import pathlib

import plotly.express as px
import polars as pl
import streamlit as st

st.set_page_config(page_title="LLM Eval Dashboard", layout="wide")
st.title("LLM Eval Harness — Results Dashboard")

reports_dir = pathlib.Path("reports")
parquet_files = list(reports_dir.glob("*.parquet"))

if not parquet_files:
    st.warning("No reports found. Run an experiment first: `make smoke`")
    st.stop()


@st.cache_data
def load_data(files: list) -> pl.DataFrame:
    frames = [pl.read_parquet(str(f)) for f in files]
    return pl.concat(frames)


df = load_data(parquet_files)

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    all_tasks = sorted(df["task"].unique().to_list())
    all_models = sorted(df["model_id"].unique().to_list())
    tasks = st.multiselect("Task", all_tasks, default=all_tasks)
    models = st.multiselect("Model", all_models, default=all_models)

filtered = df.filter(
    pl.col("task").is_in(tasks) & pl.col("model_id").is_in(models)
)

col1, col2, col3 = st.columns(3)

# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------
with col1:
    st.subheader("Scorecard")
    for model in models:
        m_df = filtered.filter(pl.col("model_id") == model)
        em = m_df["exact_match"].mean() if "exact_match" in m_df.columns else None
        f1 = m_df["token_f1"].mean() if "token_f1" in m_df.columns else None
        jv = m_df["json_valid"].mean() if "json_valid" in m_df.columns else None
        lc = m_df["label_correct"].mean() if "label_correct" in m_df.columns else None
        halt = (
            m_df["hallucination_flag"].mean()
            if "hallucination_flag" in m_df.columns
            else None
        )
        st.markdown(f"**{model}**")
        st.metric(f"{model} EM", f"{em:.2%}" if em is not None else "N/A")
        st.metric(f"{model} F1", f"{f1:.2%}" if f1 is not None else "N/A")
        st.metric(f"{model} Label Acc", f"{lc:.2%}" if lc is not None else "N/A")
        st.metric(f"{model} JSON Valid", f"{jv:.2%}" if jv is not None else "N/A")
        if halt is not None:
            st.metric(f"{model} Hallucination Rate", f"{halt:.2%}")
        st.divider()

# ---------------------------------------------------------------------------
# Accuracy vs Cost scatter
# ---------------------------------------------------------------------------
with col2:
    st.subheader("Accuracy vs Cost")
    if "exact_match" in filtered.columns and "estimated_cost_usd" in filtered.columns:
        agg = filtered.group_by("model_id").agg(
            [
                pl.col("exact_match").mean().alias("mean_em"),
                pl.col("estimated_cost_usd").mean().alias("mean_cost"),
            ]
        )
        fig = px.scatter(
            agg.to_pandas(),
            x="mean_cost",
            y="mean_em",
            text="model_id",
            labels={
                "mean_cost": "Mean Cost/Call ($)",
                "mean_em": "Mean Exact Match",
            },
            title="Accuracy vs Cost",
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No exact_match / cost data available yet.")

# ---------------------------------------------------------------------------
# Latency bars
# ---------------------------------------------------------------------------
with col3:
    st.subheader("Latency (ms)")
    if "end_to_end_ms" in filtered.columns:
        agg_lat = filtered.group_by("model_id").agg(
            [
                pl.col("end_to_end_ms").quantile(0.5).alias("p50"),
                pl.col("end_to_end_ms").quantile(0.95).alias("p95"),
            ]
        )
        melted = agg_lat.to_pandas().melt(
            id_vars="model_id", var_name="percentile", value_name="ms"
        )
        fig2 = px.bar(
            melted,
            x="model_id",
            y="ms",
            color="percentile",
            barmode="group",
            labels={"ms": "Latency (ms)"},
            title="End-to-End Latency",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No latency data available yet.")

# ---------------------------------------------------------------------------
# Failure explorer
# ---------------------------------------------------------------------------
st.subheader("Failure Explorer")
if "json_valid" in filtered.columns:
    failures = filtered.filter(pl.col("json_valid") == False)  # noqa: E712
    if failures.height > 0:
        cols_to_show = [
            c
            for c in ["example_id", "model_id", "task", "raw_text", "error"]
            if c in failures.columns
        ]
        st.dataframe(failures.select(cols_to_show).to_pandas())
    else:
        st.success("No JSON parse failures in the current selection.")
else:
    st.info("Run an experiment and generate a report to see failures.")

# ---------------------------------------------------------------------------
# Raw data table
# ---------------------------------------------------------------------------
with st.expander("Raw data"):
    st.dataframe(filtered.to_pandas())
