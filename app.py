import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import date, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import math

st.set_page_config(
    page_title="AI Task Scheduler",
    page_icon="🗓️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        h1, h2, h3 { font-family: 'Space Mono', monospace; }
        .hero {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            padding: 2.5rem 2rem; border-radius: 16px; margin-bottom: 2rem;
            color: white; text-align: center;
        }
        .hero h1 { font-size: 2.4rem; margin: 0; letter-spacing: -1px; }
        .hero p { color: #b0a8d6; font-size: 1rem; margin-top: 0.5rem; }
        .metric-card {
            background: #f5f3ff; border-left: 5px solid #7c3aed;
            padding: 1rem 1.2rem; border-radius: 10px; margin-bottom: 0.5rem;
        }
        .metric-card .label {
            font-size: 0.75rem; color: #7c3aed; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.08em;
        }
        .metric-card .value {
            font-size: 1.6rem; font-weight: 700; color: #1e1b4b;
            font-family: 'Space Mono', monospace;
        }
        .section-title {
            font-family: 'Space Mono', monospace; font-size: 1.1rem; color: #4c1d95;
            border-bottom: 2px solid #ede9fe; padding-bottom: 0.4rem; margin: 1.5rem 0 1rem;
        }
        [data-testid="stSidebar"] { background: #1e1b4b; }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stDateInput label,
        [data-testid="stSidebar"] .stNumberInput label,
        [data-testid="stSidebar"] .stTextInput label { color: #c4b5fd !important; font-weight: 500; }
        [data-testid="stSidebar"] h2 { color: white; }
        footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

PRIORITY_MAP = {"Low": 1, "Medium": 2, "High": 3}


def compute_priority_score(deadline: date, priority: str, duration: float) -> float:
    days_left = (deadline - date.today()).days
    if days_left <= 0:
        days_left = 0.1
    priority_num = PRIORITY_MAP.get(priority, 1)
    return round((5 / days_left) + (2 * priority_num) + (1 * duration), 3)


def cluster_tasks(tasks_df: pd.DataFrame) -> pd.DataFrame:
    k = min(3, len(tasks_df))
    features = tasks_df[["Days Left", "Duration (hrs)", "Priority Score"]].copy()
    features["Days Left"] = features["Days Left"].replace(0, 0.1)

    X = StandardScaler().fit_transform(features)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    center_scores = km.cluster_centers_[:, 2]
    rank = np.argsort(center_scores)[::-1]
    label_names = ["🔴 Urgent", "🟡 Moderate", "🟢 Relaxed"]
    label_map = {rank[i]: label_names[i] for i in range(k)}

    result = tasks_df.copy()
    result["Urgency Cluster"] = [label_map[l] for l in labels]
    return result


def greedy_schedule(tasks_df: pd.DataFrame, daily_hours: float) -> pd.DataFrame:
    sorted_tasks = tasks_df.sort_values("Priority Score", ascending=False).copy()
    schedule_rows = []
    current_day = date.today()
    hours_used_today = 0.0

    for _, row in sorted_tasks.iterrows():
        remaining = float(row["Duration (hrs)"])
        while remaining > 0:
            available = daily_hours - hours_used_today
            if available <= 0:
                current_day += timedelta(days=1)
                hours_used_today = 0.0
                available = daily_hours
            allocated = min(remaining, available)
            schedule_rows.append({
                "Task": row["Task Name"],
                "Date": current_day,
                "Hours Scheduled": round(allocated, 2),
                "Priority": row["Priority"],
                "Priority Score": row["Priority Score"],
                "Urgency Cluster": row.get("Urgency Cluster", "—"),
            })
            hours_used_today += allocated
            remaining -= allocated
            if math.isclose(hours_used_today, daily_hours, rel_tol=1e-5):
                current_day += timedelta(days=1)
                hours_used_today = 0.0

    return pd.DataFrame(schedule_rows)


def workload_chart(schedule_df: pd.DataFrame):
    daily = schedule_df.groupby("Date")["Hours Scheduled"].sum().reset_index().sort_values("Date")
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0f0c29")
    ax.set_facecolor("#1e1b4b")
    colors = ["#7c3aed" if i % 2 == 0 else "#a78bfa" for i in range(len(daily))]
    bars = ax.bar(daily["Date"].astype(str), daily["Hours Scheduled"],
                  color=colors, width=0.55, edgecolor="#c4b5fd", linewidth=0.6)
    for bar, val in zip(bars, daily["Hours Scheduled"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.1f}h", ha="center", va="bottom", fontsize=8, color="#e9d5ff", fontweight="bold")
    ax.set_xlabel("Date", color="#c4b5fd", fontsize=9)
    ax.set_ylabel("Total Hours", color="#c4b5fd", fontsize=9)
    ax.set_title("📊 Workload Per Day", color="white", fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(colors="#c4b5fd", labelsize=8)
    plt.xticks(rotation=35, ha="right")
    ax.spines[:].set_color("#4c1d95")
    ax.yaxis.grid(True, color="#4c1d95", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def cluster_scatter(tasks_df: pd.DataFrame):
    color_map = {"🔴 Urgent": "#ef4444", "🟡 Moderate": "#f59e0b", "🟢 Relaxed": "#22c55e"}
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0f0c29")
    ax.set_facecolor("#1e1b4b")
    for cluster, grp in tasks_df.groupby("Urgency Cluster"):
        ax.scatter(grp["Days Left"], grp["Priority Score"], label=cluster,
                   color=color_map.get(cluster, "#a78bfa"), s=110,
                   edgecolors="white", linewidths=0.6, zorder=3)
        for _, row in grp.iterrows():
            ax.annotate(row["Task Name"], (row["Days Left"], row["Priority Score"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=7, color="#e9d5ff")
    ax.set_xlabel("Days Left", color="#c4b5fd", fontsize=9)
    ax.set_ylabel("Priority Score", color="#c4b5fd", fontsize=9)
    ax.set_title("🤖 K-Means Urgency Clusters", color="white", fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(colors="#c4b5fd", labelsize=8)
    ax.spines[:].set_color("#4c1d95")
    ax.yaxis.grid(True, color="#4c1d95", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, facecolor="#302b63", edgecolor="#4c1d95", labelcolor="white")
    plt.tight_layout()
    return fig


if "tasks" not in st.session_state:
    st.session_state.tasks = []

st.markdown(
    """
    <div class="hero">
        <h1>🗓️ AI Task Scheduler</h1>
        <p>Heuristic scoring · Greedy allocation · K-Means urgency clustering</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## ➕ Add New Task")
    task_name = st.text_input("Task Name", placeholder="e.g. Write report")
    deadline = st.date_input("Deadline", min_value=date.today())
    duration = st.number_input("Estimated Duration (hours)", min_value=0.5, max_value=200.0, value=2.0, step=0.5)
    priority = st.selectbox("Manual Priority", ["Low", "Medium", "High"], index=1)
    daily_hours = st.number_input("Daily Available Hours", min_value=1.0, max_value=24.0, value=8.0, step=0.5)
    add_btn = st.button("➕ Add Task", use_container_width=True)
    st.markdown("---")
    clear_btn = st.button("🗑️ Clear All Tasks", use_container_width=True)

    if add_btn:
        if not task_name.strip():
            st.error("Please enter a task name.")
        else:
            score = compute_priority_score(deadline, priority, duration)
            days_left = (deadline - date.today()).days
            st.session_state.tasks.append({
                "Task Name": task_name.strip(),
                "Deadline": deadline,
                "Duration (hrs)": duration,
                "Priority": priority,
                "Days Left": max(days_left, 0),
                "Priority Score": score,
            })
            st.success(f"✅ **{task_name}** added! Score: **{score}**")

    if clear_btn:
        st.session_state.tasks = []
        st.info("All tasks cleared.")

if not st.session_state.tasks:
    st.info("👈 Add tasks using the sidebar to get started.")
    st.stop()

tasks_df = pd.DataFrame(st.session_state.tasks)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f'<div class="metric-card"><div class="label">Total Tasks</div>'
        f'<div class="value">{len(tasks_df)}</div></div>',
        unsafe_allow_html=True,
    )
with col2:
    total_hours = tasks_df["Duration (hrs)"].sum()
    st.markdown(
        f'<div class="metric-card"><div class="label">Total Hours</div>'
        f'<div class="value">{total_hours:.1f}h</div></div>',
        unsafe_allow_html=True,
    )
with col3:
    days_needed = math.ceil(total_hours / daily_hours)
    st.markdown(
        f'<div class="metric-card"><div class="label">Est. Days to Complete</div>'
        f'<div class="value">{days_needed} day{"s" if days_needed != 1 else ""}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown('<p class="section-title">📋 All Tasks (Input Order)</p>', unsafe_allow_html=True)
display_df = tasks_df[["Task Name", "Deadline", "Duration (hrs)", "Priority", "Days Left"]].copy()
display_df["Deadline"] = display_df["Deadline"].astype(str)
st.dataframe(display_df, use_container_width=True, hide_index=True)

st.markdown('<p class="section-title">🔢 Tasks with Computed Priority Score</p>', unsafe_allow_html=True)
scored_df = tasks_df[["Task Name", "Deadline", "Duration (hrs)", "Priority", "Days Left", "Priority Score"]].copy()
scored_df = scored_df.sort_values("Priority Score", ascending=False).reset_index(drop=True)
scored_df["Deadline"] = scored_df["Deadline"].astype(str)
st.dataframe(
    scored_df.style.background_gradient(subset=["Priority Score"], cmap="Purples"),
    use_container_width=True,
    hide_index=True,
)

tasks_df = cluster_tasks(tasks_df)

st.markdown('<p class="section-title">🤖 ML Urgency Clustering (K-Means)</p>', unsafe_allow_html=True)
st.caption("A KMeans model groups tasks into Urgent / Moderate / Relaxed based on Days Left, Duration, and Priority Score — no manual labels needed.")

cluster_display = tasks_df[["Task Name", "Days Left", "Duration (hrs)", "Priority Score", "Urgency Cluster"]].copy()
cluster_display = cluster_display.sort_values("Priority Score", ascending=False).reset_index(drop=True)
st.dataframe(cluster_display, use_container_width=True, hide_index=True)

if len(tasks_df) >= 2:
    st.pyplot(cluster_scatter(tasks_df))
else:
    st.info("Add at least 2 tasks to visualize the cluster scatter plot.")

st.markdown('<p class="section-title">📅 AI-Generated Schedule</p>', unsafe_allow_html=True)
schedule_df = greedy_schedule(tasks_df, daily_hours)
sched_display = schedule_df[["Task", "Date", "Hours Scheduled", "Priority", "Priority Score", "Urgency Cluster"]].copy()
sched_display["Date"] = sched_display["Date"].astype(str)
st.dataframe(
    sched_display.reset_index(drop=True).style.background_gradient(subset=["Hours Scheduled"], cmap="BuPu"),
    use_container_width=True,
    hide_index=True,
)

st.markdown('<p class="section-title">📊 Daily Workload Chart</p>', unsafe_allow_html=True)
st.pyplot(workload_chart(schedule_df))

with st.expander("ℹ️ How does it all work?"):
    st.markdown(
        """
        **Priority Score (heuristic)**
        ```
        priority_score = (5 / days_left) + (2 × priority_num) + (1 × duration)
        ```
        | Component | Meaning |
        |-----------|---------|
        | `5 / days_left` | Urgency — tasks due sooner score higher |
        | `2 × priority_num` | Manual priority weight (Low=1, Medium=2, High=3) |
        | `1 × duration` | Effort — longer tasks get a slight bump |

        `days_left ≤ 0` is clamped to `0.1` to avoid division by zero.

        **K-Means Urgency Clustering (ML)**

        `KMeans` from scikit-learn is fit on three scaled features: Days Left, Duration, and Priority Score. Tasks are grouped into up to 3 clusters with no manual labels. Cluster labels (Urgent / Moderate / Relaxed) are assigned by ranking centroids on their Priority Score axis.

        **Greedy Scheduler**

        Tasks sorted by Priority Score descending are allocated day-by-day up to the daily hour limit. When a day fills, scheduling continues on the next calendar day.
        """
    )
