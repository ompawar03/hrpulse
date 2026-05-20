import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime
from pathlib import Path

st.set_page_config(
    page_title="HRPulse Analytics",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

/* KPI cards */
.kpi-wrap {
    background: #ffffff;
    border: 1px solid #e4e7f0;
    border-top: 3px solid var(--accent, #2563eb);
    border-radius: 8px;
    padding: 18px 20px 14px;
    margin-bottom: 8px;
}
.kpi-label { font-size: 11px; font-weight: 500; color: #6b7280;
             letter-spacing: 0.09em; text-transform: uppercase; margin-bottom: 4px; }
.kpi-value { font-size: 26px; font-weight: 600; color: #111827;
             font-family: 'IBM Plex Mono', monospace; }
.kpi-sub   { font-size: 12px; color: #9ca3af; margin-top: 4px; }
.kpi-pos   { color: #16a34a; font-size: 12px; margin-top: 2px; }
.kpi-neg   { color: #dc2626; font-size: 12px; margin-top: 2px; }

/* Section heading */
.sec { font-size: 11px; font-weight: 600; color: #6b7280;
       letter-spacing: 0.12em; text-transform: uppercase;
       padding-bottom: 8px; border-bottom: 1px solid #e4e7f0;
       margin: 24px 0 16px; }

/* Risk badge */
.badge-low    { background:#dcfce7; color:#166534; padding:2px 10px; border-radius:12px; font-size:11px; }
.badge-medium { background:#fef9c3; color:#854d0e; padding:2px 10px; border-radius:12px; font-size:11px; }
.badge-high   { background:#fee2e2; color:#991b1b; padding:2px 10px; border-radius:12px; font-size:11px; }

div[data-testid="stSidebar"] {
    background: #f8f9fc;
    border-right: 1px solid #e4e7f0;
}
</style>
""", unsafe_allow_html=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C_BLUE   = "#2563eb"
C_TEAL   = "#0d9488"
C_AMBER  = "#d97706"
C_RED    = "#dc2626"
C_PURPLE = "#7c3aed"
C_GRAY   = "#6b7280"
PALETTE  = [C_BLUE, C_TEAL, C_AMBER, C_RED, C_PURPLE, "#059669", "#ea580c", "#0891b2"]

BG_CARD  = "#ffffff"
BG_PAGE  = "#f5f6fa"
BORDER   = "#e4e7f0"
TEXT     = "#111827"
TEXT_MUT = "#6b7280"

CHART_LAYOUT = dict(
    paper_bgcolor=BG_CARD,
    plot_bgcolor=BG_CARD,
    font=dict(family="IBM Plex Sans", color=TEXT, size=12),
    margin=dict(l=12, r=12, t=36, b=12),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
)

def cl(fig, **kw):
    fig.update_layout(**{**CHART_LAYOUT, **kw})
    return fig


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load():
    try:
        return pd.read_csv("data/hr_data.csv")
    except FileNotFoundError:
        return make_data()

def make_data(n=2500, seed=7):
    np.random.seed(seed)
    depts = ["Engineering","Product","Sales","Marketing","Finance","HR","Operations","Customer Success"]
    roles = ["Analyst","Senior Analyst","Manager","Senior Manager","Director","VP","Individual Contributor"]
    bands = ["L1","L2","L3","L4","L5","L6"]
    exits = ["Resignation","Termination","Retirement","Contract End","Transfer"]

    df = pd.DataFrame({
        "emp_id":          [f"EMP{str(i).zfill(5)}" for i in range(n)],
        "department":      np.random.choice(depts, n, p=[.22,.12,.18,.10,.08,.06,.14,.10]),
        "role":            np.random.choice(roles, n),
        "band":            np.random.choice(bands, n, p=[.18,.22,.25,.18,.10,.07]),
        "gender":          np.random.choice(["Male","Female","Non-Binary"], n, p=[.54,.43,.03]),
        "age":             np.random.randint(22, 58, n),
        "tenure_years":    np.round(np.random.exponential(3.5, n).clip(0.1, 20), 1),
        "salary":          np.round(np.random.lognormal(11.2, 0.45, n), -2),
        "performance":     np.round(np.random.beta(5, 2, n) * 4 + 1, 1).clip(1, 5),
        "satisfaction":    np.round(np.random.beta(4, 2, n) * 4 + 1, 1).clip(1, 5),
        "engagement":      np.round(np.random.beta(3, 2, n) * 4 + 1, 1).clip(1, 5),
        "leaves_taken":    np.random.randint(0, 32, n),
        "trainings":       np.random.randint(0, 9, n),
        "promotions":      np.random.randint(0, 4, n),
        "attrition":       np.random.choice([0, 1], n, p=[.82, .18]),
        "exit_reason":     np.random.choice(exits, n),
        "hire_year":       np.random.randint(2015, 2024, n),
        "remote_pct":      np.random.choice([0, 25, 50, 75, 100], n, p=[.15,.10,.30,.20,.25]),
        "overtime_hrs":    np.random.randint(0, 25, n),
    })
    df["attrition_risk"] = np.round(
        0.3 * (1 - df["satisfaction"] / 5)
      + 0.25 * (1 - df["engagement"] / 5)
      + 0.2 * (df["overtime_hrs"] / 24)
      + 0.15 * (1 - df["tenure_years"] / 20)
      + 0.1 * np.random.rand(n), 3
    ).clip(0, 1)
    df["risk_band"] = pd.cut(df["attrition_risk"],
                             bins=[0, .33, .66, 1],
                             labels=["Low", "Medium", "High"])
    df["cost_to_replace"] = np.round(df["salary"] * np.random.uniform(0.5, 2.0, n), -2)
    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👥 HRPulse")
    st.markdown("<small style='color:#9ca3af'>Workforce Intelligence v1.0</small>", unsafe_allow_html=True)
    st.divider()

    raw = load()

    st.markdown("**Filters**")
    all_depts = sorted(raw["department"].unique())
    sel_dept  = st.multiselect("Department", all_depts, default=all_depts[:5])

    all_bands = sorted(raw["band"].unique())
    sel_bands = st.multiselect("Level Band", all_bands, default=all_bands)

    all_risk  = ["Low", "Medium", "High"]
    sel_risk  = st.multiselect("Attrition Risk", all_risk, default=all_risk)

    gender_opts = sorted(raw["gender"].unique())
    sel_gender  = st.multiselect("Gender", gender_opts, default=gender_opts)

    sal_min, sal_max = int(raw["salary"].min()), int(raw["salary"].max())
    sal_range = st.slider("Salary range (₹)", sal_min, sal_max,
                          (sal_min, sal_max), step=10000,
                          format="₹%d")

    st.divider()
    st.caption("Dataset: synthetic HR records | 2,500 employees")


# ── Apply filters ─────────────────────────────────────────────────────────────
df = raw.copy()
if sel_dept:   df = df[df["department"].isin(sel_dept)]
if sel_bands:  df = df[df["band"].isin(sel_bands)]
if sel_risk:   df = df[df["risk_band"].isin(sel_risk)]
if sel_gender: df = df[df["gender"].isin(sel_gender)]
df = df[df["salary"].between(sal_range[0], sal_range[1])]


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## HRPulse Workforce Intelligence Dashboard")
st.markdown(
    f"<small style='color:#9ca3af'>{len(df):,} employees · "
    f"Refreshed {datetime.now().strftime('%d %b %Y %H:%M')}</small>",
    unsafe_allow_html=True
)
st.markdown("---")


# ── KPI Row ───────────────────────────────────────────────────────────────────
def kpi_card(col, label, val, sub="", delta=None, accent=C_BLUE, pos_good=True):
    d_html = ""
    if delta is not None:
        cls = "kpi-pos" if pos_good else "kpi-neg"
        d_html = f'<div class="{cls}">{delta}</div>'
    col.markdown(f"""
    <div class="kpi-wrap" style="--accent:{accent}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{val}</div>
      <div class="kpi-sub">{sub}</div>
      {d_html}
    </div>""", unsafe_allow_html=True)

k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
attrition_rate = df["attrition"].mean() * 100
high_risk_n    = (df["risk_band"] == "High").sum()
avg_sal        = df["salary"].mean()
avg_perf       = df["performance"].mean()
avg_sat        = df["satisfaction"].mean()
avg_eng        = df["engagement"].mean()
cost_risk      = df[df["risk_band"] == "High"]["cost_to_replace"].sum()

kpi_card(k1, "Total Headcount",   f"{len(df):,}",             "Active employees",  accent=C_BLUE)
kpi_card(k2, "Attrition Rate",    f"{attrition_rate:.1f}%",   "Historical exits",  accent=C_RED,    pos_good=False)
kpi_card(k3, "High-Risk Employees",f"{high_risk_n:,}",         "Attrition risk>66%",accent=C_AMBER,  pos_good=False)
kpi_card(k4, "Avg Salary",        f"₹{avg_sal/1e5:.1f}L",     "Annual CTC",        accent=C_TEAL)
kpi_card(k5, "Avg Performance",   f"{avg_perf:.2f}/5",        "Review score",      accent=C_PURPLE)
kpi_card(k6, "Avg Satisfaction",  f"{avg_sat:.2f}/5",         "Survey score",      accent=C_TEAL)
kpi_card(k7, "Cost at Risk",      f"₹{cost_risk/1e7:.1f}Cr",  "If high-risk leave",accent=C_RED,    pos_good=False)


# ── Row 1: Attrition by dept + Risk distribution ──────────────────────────────
st.markdown('<div class="sec">Attrition Intelligence</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns([1.6, 1, 1])

with c1:
    dept_att = (df.groupby("department")["attrition"]
                  .agg(["mean","sum","count"])
                  .reset_index()
                  .rename(columns={"mean":"rate","sum":"exits","count":"total"}))
    dept_att["rate_pct"] = (dept_att["rate"] * 100).round(1)
    dept_att = dept_att.sort_values("rate_pct", ascending=True)
    colors = [C_RED if r > 20 else C_AMBER if r > 12 else C_TEAL
              for r in dept_att["rate_pct"]]
    fig = go.Figure(go.Bar(
        x=dept_att["rate_pct"], y=dept_att["department"],
        orientation="h", marker_color=colors,
        text=dept_att["rate_pct"].map(lambda x: f"{x:.1f}%"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Rate: %{x:.1f}%<extra></extra>"
    ))
    fig.add_vline(x=dept_att["rate_pct"].mean(), line_dash="dash",
                  line_color=C_GRAY, annotation_text="Avg",
                  annotation_font_size=10)
    cl(fig, title="Attrition Rate by Department (%)", height=300,
       xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    risk_counts = df["risk_band"].value_counts().reset_index()
    risk_counts.columns = ["Risk", "Count"]
    RISK_COLORS = {"Low": C_TEAL, "Medium": C_AMBER, "High": C_RED}
    fig = px.pie(risk_counts, names="Risk", values="Count",
                 color="Risk", color_discrete_map=RISK_COLORS, hole=0.6)
    fig.update_traces(textposition="outside", textinfo="percent+label",
                      textfont_size=11, showlegend=False)
    cl(fig, title="Risk Band Distribution", height=300)
    st.plotly_chart(fig, use_container_width=True)

with c3:
    exit_data = df[df["attrition"] == 1]["exit_reason"].value_counts().reset_index()
    exit_data.columns = ["Reason", "Count"]
    fig = px.bar(exit_data, x="Count", y="Reason", orientation="h",
                 color="Count", color_continuous_scale=["#fef2f2","#dc2626"])
    fig.update_traces(showlegend=False)
    cl(fig, title="Exit Reasons (Historical)", height=300, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


# ── Row 2: Salary equity + Performance vs Satisfaction ───────────────────────
st.markdown('<div class="sec">Compensation & Performance</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    sal_gender = df.groupby(["department","gender"])["salary"].mean().reset_index()
    fig = px.bar(sal_gender, x="department", y="salary", color="gender",
                 barmode="group", color_discrete_sequence=[C_BLUE, C_TEAL, C_PURPLE],
                 labels={"salary":"Avg Salary (₹)","department":""})
    fig.update_layout(xaxis_tickangle=-30)
    cl(fig, title="Salary Distribution by Dept & Gender", height=300)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.scatter(
        df.sample(min(600, len(df)), random_state=1),
        x="satisfaction", y="performance",
        color="risk_band", size="salary",
        color_discrete_map={"Low":C_TEAL,"Medium":C_AMBER,"High":C_RED},
        opacity=0.65, size_max=18,
        labels={"satisfaction":"Satisfaction Score","performance":"Performance Score"},
        hover_data=["department","band","tenure_years"]
    )
    cl(fig, title="Performance vs Satisfaction (bubble=salary, color=risk)", height=300)
    st.plotly_chart(fig, use_container_width=True)


# ── Row 3: Tenure heatmap + Hiring trend ─────────────────────────────────────
st.markdown('<div class="sec">Workforce Demographics</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    age_bins  = pd.cut(df["age"], bins=[21,30,35,40,45,50,60],
                       labels=["22-30","31-35","36-40","41-45","46-50","51-60"])
    heat_data = (df.assign(age_group=age_bins)
                   .groupby(["department","age_group"])["emp_id"].count()
                   .unstack(fill_value=0))
    fig = px.imshow(heat_data,
                    color_continuous_scale=["#eff6ff","#2563eb"],
                    aspect="auto", text_auto=True)
    cl(fig, title="Headcount Heatmap: Department × Age Group", height=310)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    hire_trend = df.groupby(["hire_year","department"])["emp_id"].count().reset_index()
    hire_trend.columns = ["year","department","hires"]
    top_depts  = df["department"].value_counts().head(5).index.tolist()
    hire_trend = hire_trend[hire_trend["department"].isin(top_depts)]
    fig = px.line(hire_trend, x="year", y="hires", color="department",
                  color_discrete_sequence=PALETTE, markers=True,
                  labels={"hires":"New Hires","year":"Hire Year","department":""})
    cl(fig, title="Annual Hiring Trend by Department", height=310)
    st.plotly_chart(fig, use_container_width=True)


# ── Row 4: Engagement vs overtime + Remote work ───────────────────────────────
st.markdown('<div class="sec">Engagement & Work Patterns</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

with c1:
    ot_bins = pd.cut(df["overtime_hrs"], bins=[-1,4,10,17,25],
                     labels=["0–4h","5–10h","11–17h","18–25h"])
    ot_eng  = df.assign(ot_band=ot_bins).groupby("ot_band")["engagement"].mean().reset_index()
    fig = px.bar(ot_eng, x="ot_band", y="engagement",
                 color="engagement", color_continuous_scale=["#dc2626","#16a34a"],
                 labels={"ot_band":"Overtime Band","engagement":"Avg Engagement"})
    fig.update_traces(showlegend=False)
    cl(fig, title="Engagement vs Overtime", height=270, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    remote_att = df.groupby("remote_pct")["attrition"].mean().reset_index()
    fig = px.line(remote_att, x="remote_pct", y="attrition",
                  markers=True, line_shape="spline",
                  labels={"remote_pct":"Remote Work %","attrition":"Attrition Rate"})
    fig.update_traces(line_color=C_BLUE, marker_color=C_RED)
    cl(fig, title="Attrition Rate vs Remote Work %", height=270)
    st.plotly_chart(fig, use_container_width=True)

with c3:
    promo_att = df.groupby("promotions")["attrition"].mean().reset_index()
    fig = px.bar(promo_att, x="promotions", y="attrition",
                 color="attrition",
                 color_continuous_scale=["#dcfce7","#dc2626"],
                 labels={"promotions":"Promotions Received","attrition":"Attrition Rate"})
    fig.update_traces(showlegend=False)
    cl(fig, title="Attrition Rate by Promotions Received", height=270,
       coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


# ── Row 5: High-risk employee table + Export ──────────────────────────────────
st.markdown('<div class="sec">High-Risk Employee Watchlist</div>', unsafe_allow_html=True)

high_risk_df = (df[df["risk_band"] == "High"]
                  [["emp_id","department","role","band","tenure_years",
                     "salary","performance","satisfaction","engagement",
                     "overtime_hrs","attrition_risk","cost_to_replace"]]
                  .sort_values("attrition_risk", ascending=False)
                  .head(50))

st.dataframe(
    high_risk_df.style
      .background_gradient(subset=["attrition_risk"], cmap="Reds")
      .background_gradient(subset=["cost_to_replace"], cmap="YlOrRd")
      .background_gradient(subset=["performance"], cmap="Greens")
      .format({
          "salary": "₹{:,.0f}",
          "cost_to_replace": "₹{:,.0f}",
          "attrition_risk": "{:.3f}",
          "tenure_years": "{:.1f}y"
      }),
    use_container_width=True, height=340
)

dl_col, _ = st.columns([1, 5])
with dl_col:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(
        "⬇ Export Full Dataset",
        buf.getvalue(),
        file_name=f"hrpulse_export_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("HRPulse Workforce Intelligence · Python · Pandas · Plotly · Streamlit · © 2025")