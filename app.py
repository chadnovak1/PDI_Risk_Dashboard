import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="PDI Risk Dashboard", layout="wide")

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = Path("data")

REQUIRED_FILES = {
    "loss_runs": "loss_runs.csv",
    "payroll": "payroll_hours.csv",
    "fleet": "fleet_events.csv",
    "incidents": "incident_reports.csv",
    "utility": "utility_strikes.csv",
}

RISK_WEIGHTS = {
    "utility_strike": 40,
    "utility_strike_tolerance_zone": 25,
    "missing_locate": 25,
    "excavation_near_miss": 10,
    "safety_violation": 5,
    "vehicle_accident": 30,
    "speeding_per_10": 5,
    "harsh_braking_per_10": 4,
    "harsh_accel_per_10": 3,
    "seatbelt_violation": 10,
    "distracted_driving": 15,
    "recordable_injury": 25,
    "lost_time_injury": 40,
    "auto_liability_claim": 35,
    "property_damage_claim": 15,
}


# -----------------------------
# Helper Functions
# -----------------------------
def safe_read_csv(path: Path, columns=None):
    if not path.exists():
        return pd.DataFrame(columns=columns or [])
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return pd.DataFrame(columns=columns or [])


def normalize_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def parse_dates(df: pd.DataFrame, date_cols):
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def ensure_columns(df: pd.DataFrame, defaults: dict):
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    return df


def classify_risk(score: float) -> str:
    if pd.isna(score):
        return "Unknown"
    if score <= 20:
        return "Low"
    if score <= 40:
        return "Moderate"
    if score <= 60:
        return "High"
    return "Severe"


def risk_color(level: str) -> str:
    return {
        "Low": "#2e7d32",
        "Moderate": "#f9a825",
        "High": "#ef6c00",
        "Severe": "#c62828",
        "Unknown": "#616161",
    }.get(level, "#616161")


def load_data():
    loss = safe_read_csv(
        DATA_DIR / REQUIRED_FILES["loss_runs"],
        [
            "ClaimNumber", "ClaimType", "IncidentDate", "ReportDate", "Crew", "Supervisor",
            "Cause", "BodyPart", "Paid", "Reserved", "TotalIncurred", "Status",
            "Recordable", "LostTime"
        ],
    )
    payroll = safe_read_csv(
        DATA_DIR / REQUIRED_FILES["payroll"],
        ["Year", "Month", "Crew", "HoursWorked", "Payroll", "Employees"],
    )
    fleet = safe_read_csv(
        DATA_DIR / REQUIRED_FILES["fleet"],
        [
            "Date", "Vehicle", "Driver", "Crew", "Miles", "SpeedingEvents", "HarshBrake",
            "HarshAccel", "SeatbeltViolations", "DistractedDrivingEvents", "VehicleAccidents"
        ],
    )
    incidents = safe_read_csv(
        DATA_DIR / REQUIRED_FILES["incidents"],
        [
            "Date", "IncidentType", "Crew", "Supervisor", "Equipment", "Description",
            "NearMiss", "RecordableInjury", "LostTimeInjury", "SafetyViolation"
        ],
    )
    utility = safe_read_csv(
        DATA_DIR / REQUIRED_FILES["utility"],
        [
            "Date", "Crew", "UtilityType", "Located", "InsideToleranceZone", "RepairCost",
            "MissingLocate", "Notes"
        ],
    )

    loss = parse_dates(loss, ["IncidentDate", "ReportDate"])
    payroll = ensure_columns(payroll, {"HoursWorked": 0, "Payroll": 0, "Employees": 0})
    fleet = parse_dates(fleet, ["Date"])
    incidents = parse_dates(incidents, ["Date"])
    utility = parse_dates(utility, ["Date"])

    for df, col in [
        (loss, "Crew"), (payroll, "Crew"), (fleet, "Crew"), (incidents, "Crew"), (utility, "Crew")
    ]:
        if col in df.columns:
            df[col] = normalize_text(df[col])

    numeric_defaults = {
        "Paid": 0, "Reserved": 0, "TotalIncurred": 0,
        "Miles": 0, "SpeedingEvents": 0, "HarshBrake": 0, "HarshAccel": 0,
        "SeatbeltViolations": 0, "DistractedDrivingEvents": 0, "VehicleAccidents": 0,
        "NearMiss": 0, "RecordableInjury": 0, "LostTimeInjury": 0, "SafetyViolation": 0,
        "RepairCost": 0, "MissingLocate": 0,
    }
    for df in [loss, fleet, incidents, utility]:
        for col, default in numeric_defaults.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    if "HoursWorked" in payroll.columns:
        payroll["HoursWorked"] = pd.to_numeric(payroll["HoursWorked"], errors="coerce").fillna(0)
    if "Payroll" in payroll.columns:
        payroll["Payroll"] = pd.to_numeric(payroll["Payroll"], errors="coerce").fillna(0)

    return loss, payroll, fleet, incidents, utility


def build_crew_risk_table(loss, payroll, fleet, incidents, utility):
    hours = payroll.groupby("Crew", dropna=False)["HoursWorked"].sum().rename("HoursWorked")

    operational = pd.DataFrame(index=hours.index)
    if not utility.empty:
        temp = utility.groupby("Crew").agg(
            utility_strikes=("UtilityType", "count"),
            inside_tolerance=("InsideToleranceZone", lambda s: pd.to_numeric(s, errors='coerce').fillna(0).sum()),
            missing_locate=("MissingLocate", "sum"),
            repair_cost=("RepairCost", "sum"),
        )
        operational = operational.join(temp, how="outer")
    if not incidents.empty:
        near = incidents.groupby("Crew").agg(
            excavation_near_miss=("NearMiss", "sum"),
            safety_violations=("SafetyViolation", "sum"),
        )
        operational = operational.join(near, how="outer")

    operational = operational.fillna(0)
    operational["OperationalPoints"] = (
        operational.get("utility_strikes", 0) * RISK_WEIGHTS["utility_strike"] +
        operational.get("inside_tolerance", 0) * RISK_WEIGHTS["utility_strike_tolerance_zone"] +
        operational.get("missing_locate", 0) * RISK_WEIGHTS["missing_locate"] +
        operational.get("excavation_near_miss", 0) * RISK_WEIGHTS["excavation_near_miss"] +
        operational.get("safety_violations", 0) * RISK_WEIGHTS["safety_violation"]
    )

    fleet_agg = pd.DataFrame(index=hours.index)
    if not fleet.empty:
        temp = fleet.groupby("Crew").agg(
            Miles=("Miles", "sum"),
            SpeedingEvents=("SpeedingEvents", "sum"),
            HarshBrake=("HarshBrake", "sum"),
            HarshAccel=("HarshAccel", "sum"),
            SeatbeltViolations=("SeatbeltViolations", "sum"),
            DistractedDrivingEvents=("DistractedDrivingEvents", "sum"),
            VehicleAccidents=("VehicleAccidents", "sum"),
        )
        fleet_agg = fleet_agg.join(temp, how="outer")
    fleet_agg = fleet_agg.fillna(0)
    fleet_agg["FleetPoints"] = (
        fleet_agg.get("VehicleAccidents", 0) * RISK_WEIGHTS["vehicle_accident"] +
        np.floor(fleet_agg.get("SpeedingEvents", 0) / 10) * RISK_WEIGHTS["speeding_per_10"] +
        np.floor(fleet_agg.get("HarshBrake", 0) / 10) * RISK_WEIGHTS["harsh_braking_per_10"] +
        np.floor(fleet_agg.get("HarshAccel", 0) / 10) * RISK_WEIGHTS["harsh_accel_per_10"] +
        fleet_agg.get("SeatbeltViolations", 0) * RISK_WEIGHTS["seatbelt_violation"] +
        fleet_agg.get("DistractedDrivingEvents", 0) * RISK_WEIGHTS["distracted_driving"]
    )

    claim_agg = pd.DataFrame(index=hours.index)
    if not loss.empty:
        loss = ensure_columns(loss, {"Recordable": 0, "LostTime": 0, "ClaimType": "Unknown"})
        temp = loss.groupby("Crew").agg(
            ClaimCount=("ClaimNumber", "count"),
            TotalIncurred=("TotalIncurred", "sum"),
            RecordableClaims=("Recordable", "sum"),
            LostTimeClaims=("LostTime", "sum"),
            AutoClaims=("ClaimType", lambda s: (s.astype(str).str.contains("auto", case=False, na=False)).sum()),
            PropertyDamageClaims=("ClaimType", lambda s: (s.astype(str).str.contains("property", case=False, na=False)).sum()),
        )
        claim_agg = claim_agg.join(temp, how="outer")
    claim_agg = claim_agg.fillna(0)
    claim_agg["ClaimPoints"] = (
        claim_agg.get("RecordableClaims", 0) * RISK_WEIGHTS["recordable_injury"] +
        claim_agg.get("LostTimeClaims", 0) * RISK_WEIGHTS["lost_time_injury"] +
        claim_agg.get("AutoClaims", 0) * RISK_WEIGHTS["auto_liability_claim"] +
        claim_agg.get("PropertyDamageClaims", 0) * RISK_WEIGHTS["property_damage_claim"]
    )

    df = pd.concat([hours, operational, fleet_agg, claim_agg], axis=1).fillna(0)
    df = df.reset_index()
    df["OperationalScore"] = np.where(df["HoursWorked"] > 0, (df["OperationalPoints"] / df["HoursWorked"]) * 10000, 0)
    df["ClaimScore"] = np.where(df["HoursWorked"] > 0, (df["ClaimPoints"] / df["HoursWorked"]) * 10000, 0)
    df["FleetScore"] = np.where(df["Miles"] > 0, (df["FleetPoints"] / df["Miles"]) * 10000, 0)
    df["TotalRiskScore"] = df[["OperationalScore", "FleetScore", "ClaimScore"]].sum(axis=1)
    df["RiskLevel"] = df["TotalRiskScore"].apply(classify_risk)
    return df.sort_values("TotalRiskScore", ascending=False)


def monthly_claim_trend(loss):
    if loss.empty or "IncidentDate" not in loss.columns:
        return pd.DataFrame(columns=["Month", "Claims", "Cost"])
    df = loss.dropna(subset=["IncidentDate"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["Month", "Claims", "Cost"])
    df["Month"] = df["IncidentDate"].dt.to_period("M").astype(str)
    out = df.groupby("Month").agg(Claims=("ClaimNumber", "count"), Cost=("TotalIncurred", "sum")).reset_index()
    return out


def monthly_fleet_trend(fleet):
    if fleet.empty or "Date" not in fleet.columns:
        return pd.DataFrame(columns=["Month", "SpeedingEvents", "HarshBrake", "VehicleAccidents"])
    df = fleet.dropna(subset=["Date"]).copy()
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    return df.groupby("Month").agg(
        SpeedingEvents=("SpeedingEvents", "sum"),
        HarshBrake=("HarshBrake", "sum"),
        VehicleAccidents=("VehicleAccidents", "sum"),
    ).reset_index()


def monthly_utility_trend(utility):
    if utility.empty or "Date" not in utility.columns:
        return pd.DataFrame(columns=["Month", "Strikes", "RepairCost"])
    df = utility.dropna(subset=["Date"]).copy()
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    return df.groupby("Month").agg(
        Strikes=("UtilityType", "count"),
        RepairCost=("RepairCost", "sum"),
    ).reset_index()


# -----------------------------
# Data Load
# -----------------------------
loss, payroll, fleet, incidents, utility = load_data()
crew_risk = build_crew_risk_table(loss, payroll, fleet, incidents, utility)
claim_trend = monthly_claim_trend(loss)
fleet_trend = monthly_fleet_trend(fleet)
utility_trend = monthly_utility_trend(utility)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("PDI Risk Dashboard")
page = st.sidebar.radio(
    "Select View",
    [
        "Executive Overview",
        "Crew Risk Ranking",
        "Claims Analytics",
        "Fleet Risk",
        "Utility Strike Tracker",
        "Data Health",
    ],
)

if not crew_risk.empty:
    crews = ["All"] + sorted([c for c in crew_risk["Crew"].dropna().unique() if str(c).strip()])
else:
    crews = ["All"]
selected_crew = st.sidebar.selectbox("Crew Filter", crews)

if selected_crew != "All":
    crew_risk_view = crew_risk[crew_risk["Crew"] == selected_crew]
    loss_view = loss[loss["Crew"] == selected_crew] if "Crew" in loss.columns else loss
    fleet_view = fleet[fleet["Crew"] == selected_crew] if "Crew" in fleet.columns else fleet
    utility_view = utility[utility["Crew"] == selected_crew] if "Crew" in utility.columns else utility
else:
    crew_risk_view = crew_risk
    loss_view = loss
    fleet_view = fleet
    utility_view = utility

# -----------------------------
# Header
# -----------------------------
st.title("PDI Construction Risk Dashboard")
st.caption("Construction risk dashboard for claims, fleet, utility strikes, and crew risk scoring.")

# -----------------------------
# Executive Overview
# -----------------------------
if page == "Executive Overview":
    total_claims = len(loss_view)
    total_cost = pd.to_numeric(loss_view.get("TotalIncurred", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    total_strikes = len(utility_view)
    total_vehicle_accidents = pd.to_numeric(fleet_view.get("VehicleAccidents", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    avg_risk = crew_risk_view["TotalRiskScore"].mean() if not crew_risk_view.empty else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Claims", f"{int(total_claims)}")
    c2.metric("Total Claim Cost", f"${total_cost:,.0f}")
    c3.metric("Utility Strikes", f"{int(total_strikes)}")
    c4.metric("Vehicle Accidents", f"{int(total_vehicle_accidents)}")
    c5.metric("Avg Crew Risk Score", f"{avg_risk:,.1f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Claim Cost Trend")
        if not claim_trend.empty:
            fig = px.bar(claim_trend, x="Month", y="Cost")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No claim trend data available yet.")

    with right:
        st.subheader("Utility Strike Trend")
        if not utility_trend.empty:
            fig = px.line(utility_trend, x="Month", y="Strikes", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No utility strike data available yet.")

    st.subheader("Crew Risk Summary")
    if not crew_risk_view.empty:
        show = crew_risk_view[["Crew", "HoursWorked", "OperationalScore", "FleetScore", "ClaimScore", "TotalRiskScore", "RiskLevel"]].copy()
        st.dataframe(show, use_container_width=True)
    else:
        st.info("No crew risk data available yet.")

# -----------------------------
# Crew Risk Ranking
# -----------------------------
elif page == "Crew Risk Ranking":
    st.subheader("Crew Risk Ranking")
    if crew_risk_view.empty:
        st.info("No crew risk data available yet.")
    else:
        fig = px.bar(
            crew_risk_view,
            x="Crew",
            y="TotalRiskScore",
            color="RiskLevel",
            color_discrete_map={
                "Low": "#2e7d32",
                "Moderate": "#f9a825",
                "High": "#ef6c00",
                "Severe": "#c62828",
            },
            hover_data=["OperationalScore", "FleetScore", "ClaimScore", "HoursWorked"],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            crew_risk_view[[
                "Crew", "HoursWorked", "Miles", "OperationalPoints", "FleetPoints", "ClaimPoints",
                "OperationalScore", "FleetScore", "ClaimScore", "TotalRiskScore", "RiskLevel"
            ]],
            use_container_width=True,
        )

# -----------------------------
# Claims Analytics
# -----------------------------
elif page == "Claims Analytics":
    st.subheader("Claims Analytics")
    if loss_view.empty:
        st.info("No loss run data available yet.")
    else:
        top1, top2 = st.columns(2)
        with top1:
            if "ClaimType" in loss_view.columns:
                claim_type = loss_view.groupby("ClaimType").size().reset_index(name="Claims")
                fig = px.pie(claim_type, names="ClaimType", values="Claims", title="Claims by Type")
                st.plotly_chart(fig, use_container_width=True)
        with top2:
            if "Cause" in loss_view.columns:
                cause = loss_view.groupby("Cause").size().reset_index(name="Claims").sort_values("Claims", ascending=False).head(10)
                fig = px.bar(cause, x="Claims", y="Cause", orientation="h", title="Top Claim Causes")
                st.plotly_chart(fig, use_container_width=True)

        if {"IncidentDate", "ReportDate"}.issubset(loss_view.columns):
            lag = loss_view.copy()
            lag["LagDays"] = (lag["ReportDate"] - lag["IncidentDate"]).dt.days
            lag = lag.dropna(subset=["LagDays"])
            if not lag.empty:
                st.subheader("Claim Reporting Lag")
                st.metric("Average Lag Days", f"{lag['LagDays'].mean():.1f}")
                fig = px.histogram(lag, x="LagDays", nbins=20)
                st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Fleet Risk
# -----------------------------
elif page == "Fleet Risk":
    st.subheader("Fleet Risk")
    if fleet_view.empty:
        st.info("No fleet data available yet.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Miles Driven", f"{fleet_view['Miles'].sum():,.0f}")
        c2.metric("Speeding Events", f"{fleet_view['SpeedingEvents'].sum():,.0f}")
        c3.metric("Vehicle Accidents", f"{fleet_view['VehicleAccidents'].sum():,.0f}")

        if not fleet_trend.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fleet_trend["Month"], y=fleet_trend["SpeedingEvents"], name="Speeding"))
            fig.add_trace(go.Scatter(x=fleet_trend["Month"], y=fleet_trend["HarshBrake"], name="Harsh Braking"))
            st.plotly_chart(fig, use_container_width=True)

        by_driver = fleet_view.groupby("Driver", dropna=False).agg(
            Miles=("Miles", "sum"),
            Speeding=("SpeedingEvents", "sum"),
            HarshBrake=("HarshBrake", "sum"),
            HarshAccel=("HarshAccel", "sum"),
            Accidents=("VehicleAccidents", "sum"),
        ).reset_index().sort_values("Speeding", ascending=False)
        st.subheader("Driver Event Summary")
        st.dataframe(by_driver, use_container_width=True)

# -----------------------------
# Utility Strike Tracker
# -----------------------------
elif page == "Utility Strike Tracker":
    st.subheader("Utility Strike Tracker")
    if utility_view.empty:
        st.info("No utility strike data available yet.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Strikes", f"{len(utility_view):,}")
        c2.metric("Total Repair Cost", f"${utility_view['RepairCost'].sum():,.0f}")
        c3.metric("Avg Repair Cost", f"${utility_view['RepairCost'].mean():,.0f}" if len(utility_view) else "$0")

        by_type = utility_view.groupby("UtilityType").size().reset_index(name="Strikes")
        fig = px.bar(by_type, x="UtilityType", y="Strikes", title="Strikes by Utility Type")
        st.plotly_chart(fig, use_container_width=True)

        by_crew = utility_view.groupby("Crew").agg(Strikes=("UtilityType", "count"), RepairCost=("RepairCost", "sum")).reset_index()
        fig = px.bar(by_crew, x="Crew", y="Strikes", hover_data=["RepairCost"], title="Strikes by Crew")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(utility_view, use_container_width=True)

# -----------------------------
# Data Health
# -----------------------------
else:
    st.subheader("Data Health Check")
    health = pd.DataFrame(
        [
            {"Dataset": "Loss Runs", "Rows": len(loss), "File": REQUIRED_FILES["loss_runs"], "Exists": (DATA_DIR / REQUIRED_FILES["loss_runs"]).exists()},
            {"Dataset": "Payroll / Hours", "Rows": len(payroll), "File": REQUIRED_FILES["payroll"], "Exists": (DATA_DIR / REQUIRED_FILES["payroll"]).exists()},
            {"Dataset": "Fleet Events", "Rows": len(fleet), "File": REQUIRED_FILES["fleet"], "Exists": (DATA_DIR / REQUIRED_FILES["fleet"]).exists()},
            {"Dataset": "Incident Reports", "Rows": len(incidents), "File": REQUIRED_FILES["incidents"], "Exists": (DATA_DIR / REQUIRED_FILES["incidents"]).exists()},
            {"Dataset": "Utility Strikes", "Rows": len(utility), "File": REQUIRED_FILES["utility"], "Exists": (DATA_DIR / REQUIRED_FILES["utility"]).exists()},
        ]
    )
    st.dataframe(health, use_container_width=True)

    st.markdown("### Expected Repository Structure")
    st.code(
        """PDI_Risk_Dashboard/
├── app.py
├── requirements.txt
└── data/
    ├── loss_runs.csv
    ├── payroll_hours.csv
    ├── fleet_events.csv
    ├── incident_reports.csv
    └── utility_strikes.csv
"""
    )

    st.markdown("### requirements.txt")
    st.code(
        """streamlit
pandas
numpy
plotly
"""
    )
