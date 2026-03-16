import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="PDI Risk Dashboard", layout="wide")

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DIR = BASE_DIR / "sample_data"

FILES = {
    "loss_runs": "loss_runs_sample.csv",
    "payroll": "payroll_hours_sample.csv",
    "fleet": "fleet_events_sample.csv",
    "incidents": "incident_reports_sample.csv",
    "utility": "utility_strikes_sample.csv",
}

# ============================================================
# HELPERS
# ============================================================
def load_csv(filename: str) -> pd.DataFrame:
    """Load from /data first, then fall back to /sample_data."""
    data_file = DATA_DIR / filename.replace("_sample", "")
    sample_file = SAMPLE_DIR / filename

    if data_file.exists():
        return pd.read_csv(data_file)
    if sample_file.exists():
        return pd.read_csv(sample_file)

    return pd.DataFrame()


def yes_no_to_int(series: pd.Series) -> pd.Series:
    """Convert Yes/No style fields to 1/0."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})
        .fillna(0)
        .astype(int)
    )


def to_numeric_safe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def to_datetime_safe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def risk_level(score: float) -> str:
    if score <= 20:
        return "Low"
    elif score <= 40:
        return "Moderate"
    elif score <= 60:
        return "High"
    else:
        return "Severe"


# ============================================================
# LOAD DATA
# ============================================================
loss_df = load_csv(FILES["loss_runs"])
payroll_df = load_csv(FILES["payroll"])
fleet_df = load_csv(FILES["fleet"])
incident_df = load_csv(FILES["incidents"])
utility_df = load_csv(FILES["utility"])

# ============================================================
# EXPECTED COLUMNS
# ============================================================
LOSS_COLS = [
    "ClaimNumber", "ClaimType", "IncidentDate", "ReportDate", "Crew",
    "Supervisor", "EmployeeName", "BodyPart", "Cause", "Paid",
    "Reserved", "TotalIncurred", "Recordable", "LostTime", "Status"
]

PAYROLL_COLS = [
    "Year", "Month", "Crew", "HoursWorked", "Employees", "Payroll"
]

FLEET_COLS = [
    "Date", "VehicleID", "Driver", "JobTitle", "Crew", "MilesDriven",
    "EngineHours", "SpeedingEvents", "HarshBrakeEvents",
    "HarshAccelEvents", "SeatbeltViolations",
    "DistractedDrivingEvents", "VehicleAccidents", "IdleHours"
]

INCIDENT_COLS = [
    "IncidentID", "Date", "Time", "Crew", "Supervisor", "EmployeeName",
    "IncidentType", "EquipmentInvolved", "CitationIssued",
    "SafetyViolation", "Location", "HiringContractor"
]

UTILITY_COLS = [
    "StrikeID", "Date", "Time", "Crew", "JobTitle", "Supervisor",
    "EquipmentUsed", "UtilityType", "Located", "ToleranceZone",
    "RepairCost", "CitationIssued", "Location", "HiringContractor"
]

# Ensure missing dataframes still have expected headers
if loss_df.empty:
    loss_df = pd.DataFrame(columns=LOSS_COLS)

if payroll_df.empty:
    payroll_df = pd.DataFrame(columns=PAYROLL_COLS)

if fleet_df.empty:
    fleet_df = pd.DataFrame(columns=FLEET_COLS)

if incident_df.empty:
    incident_df = pd.DataFrame(columns=INCIDENT_COLS)

if utility_df.empty:
    utility_df = pd.DataFrame(columns=UTILITY_COLS)

# ============================================================
# CLEAN / NORMALIZE DATA
# ============================================================
loss_df = to_datetime_safe(loss_df, ["IncidentDate", "ReportDate"])
payroll_df = to_numeric_safe(payroll_df, ["HoursWorked", "Employees", "Payroll"])
fleet_df = to_datetime_safe(fleet_df, ["Date"])
fleet_df = to_numeric_safe(
    fleet_df,
    [
        "MilesDriven", "EngineHours", "SpeedingEvents", "HarshBrakeEvents",
        "HarshAccelEvents", "SeatbeltViolations",
        "DistractedDrivingEvents", "VehicleAccidents", "IdleHours"
    ],
)
incident_df = to_datetime_safe(incident_df, ["Date"])
utility_df = to_datetime_safe(utility_df, ["Date"])
utility_df = to_numeric_safe(utility_df, ["RepairCost"])

if "Recordable" in loss_df.columns:
    loss_df["Recordable"] = yes_no_to_int(loss_df["Recordable"])

if "LostTime" in loss_df.columns:
    loss_df["LostTime"] = yes_no_to_int(loss_df["LostTime"])

if "CitationIssued" in incident_df.columns:
    incident_df["CitationIssued_Flag"] = yes_no_to_int(incident_df["CitationIssued"])
else:
    incident_df["CitationIssued_Flag"] = 0

if "SafetyViolation" in incident_df.columns:
    incident_df["SafetyViolation_Flag"] = yes_no_to_int(incident_df["SafetyViolation"])
else:
    incident_df["SafetyViolation_Flag"] = 0

if "Located" in utility_df.columns:
    utility_df["Located_Flag"] = yes_no_to_int(utility_df["Located"])
else:
    utility_df["Located_Flag"] = 0

if "ToleranceZone" in utility_df.columns:
    utility_df["ToleranceZone_Flag"] = yes_no_to_int(utility_df["ToleranceZone"])
else:
    utility_df["ToleranceZone_Flag"] = 0

if "CitationIssued" in utility_df.columns:
    utility_df["CitationIssued_Flag"] = yes_no_to_int(utility_df["CitationIssued"])
else:
    utility_df["CitationIssued_Flag"] = 0

# Fill blanks for grouping
for df, col in [
    (loss_df, "Crew"),
    (payroll_df, "Crew"),
    (fleet_df, "Crew"),
    (incident_df, "Crew"),
    (utility_df, "Crew"),
]:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

# ============================================================
# RISK MODEL
# ============================================================
def build_crew_risk_table():
    crew_hours = payroll_df.groupby("Crew", dropna=False)["HoursWorked"].sum().reset_index()
    crew_hours.columns = ["Crew", "HoursWorked"]

    # Operational risk from incidents + utility strikes
    incident_counts = incident_df.groupby(["Crew", "IncidentType"]).size().unstack(fill_value=0).reset_index()
    utility_counts = utility_df.groupby("Crew").agg(
        UtilityStrikes=("StrikeID", "count"),
        UtilityStrikeCost=("RepairCost", "sum"),
        MissingLocates=("Located_Flag", lambda s: int((s == 0).sum())),
        ToleranceZoneHits=("ToleranceZone_Flag", "sum"),
    ).reset_index()

    # Fleet risk
    fleet_agg = fleet_df.groupby("Crew").agg(
        MilesDriven=("MilesDriven", "sum"),
        SpeedingEvents=("SpeedingEvents", "sum"),
        HarshBrakeEvents=("HarshBrakeEvents", "sum"),
        HarshAccelEvents=("HarshAccelEvents", "sum"),
        SeatbeltViolations=("SeatbeltViolations", "sum"),
        DistractedDrivingEvents=("DistractedDrivingEvents", "sum"),
        VehicleAccidents=("VehicleAccidents", "sum"),
    ).reset_index()

    # Claim risk
    claims_agg = loss_df.groupby("Crew").agg(
        ClaimCount=("ClaimNumber", "count"),
        TotalIncurred=("TotalIncurred", "sum"),
        RecordableClaims=("Recordable", "sum"),
        LostTimeClaims=("LostTime", "sum"),
    ).reset_index()

    # Merge all
    crew_risk = crew_hours.copy()
    for part in [incident_counts, utility_counts, fleet_agg, claims_agg]:
        crew_risk = crew_risk.merge(part, on="Crew", how="left")

    crew_risk = crew_risk.fillna(0)

    # Make sure these columns exist even if no data
    for col in [
        "Near Miss",
        "Recordable Injury",
        "Auto Accident",
        "Utility Strike",
        "Property Damage",
        "Safety Violation",
    ]:
        if col not in crew_risk.columns:
            crew_risk[col] = 0

    # Operational points
    crew_risk["OperationalPoints"] = (
        crew_risk["UtilityStrikes"] * 40
        + crew_risk["MissingLocates"] * 25
        + crew_risk["ToleranceZoneHits"] * 25
        + crew_risk["Near Miss"] * 10
        + crew_risk["Safety Violation"] * 5
        + crew_risk["Recordable Injury"] * 20
    )

    # Fleet points
    crew_risk["FleetPoints"] = (
        crew_risk["VehicleAccidents"] * 30
        + (crew_risk["SpeedingEvents"] / 10.0) * 5
        + (crew_risk["HarshBrakeEvents"] / 10.0) * 4
        + (crew_risk["HarshAccelEvents"] / 10.0) * 3
        + crew_risk["SeatbeltViolations"] * 10
        + crew_risk["DistractedDrivingEvents"] * 15
    )

    # Claim points
    crew_risk["ClaimPoints"] = (
        crew_risk["RecordableClaims"] * 25
        + crew_risk["LostTimeClaims"] * 40
    )

    # Normalize
    crew_risk["OperationalScore"] = np.where(
        crew_risk["HoursWorked"] > 0,
        (crew_risk["OperationalPoints"] / crew_risk["HoursWorked"]) * 10000,
        0,
    )

    crew_risk["ClaimScore"] = np.where(
        crew_risk["HoursWorked"] > 0,
        (crew_risk["ClaimPoints"] / crew_risk["HoursWorked"]) * 10000,
        0,
    )

    crew_risk["FleetScore"] = np.where(
        crew_risk["MilesDriven"] > 0,
        (crew_risk["FleetPoints"] / crew_risk["MilesDriven"]) * 10000,
        0,
    )

    crew_risk["TotalRiskScore"] = (
        crew_risk["OperationalScore"] + crew_risk["FleetScore"] + crew_risk["ClaimScore"]
    )

    crew_risk["RiskLevel"] = crew_risk["TotalRiskScore"].apply(risk_level)

    return crew_risk.sort_values("TotalRiskScore", ascending=False)


crew_risk_df = build_crew_risk_table()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Executive Overview",
        "Crew Risk Ranking",
        "Claims Analytics",
        "Fleet Risk",
        "Utility Strike Tracker",
        "Data Health",
    ],
)

# ============================================================
# HEADER
# ============================================================
st.title("PDI Risk Dashboard")
st.caption("Claims, fleet, utility strike, and incident analytics for PDI Construction")

# ============================================================
# PAGE: EXECUTIVE OVERVIEW
# ============================================================
if page == "Executive Overview":
    total_claims = len(loss_df)
    total_claim_cost = loss_df["TotalIncurred"].sum() if "TotalIncurred" in loss_df.columns else 0
    total_strikes = len(utility_df)
    total_vehicle_accidents = fleet_df["VehicleAccidents"].sum() if "VehicleAccidents" in fleet_df.columns else 0
    avg_risk_score = crew_risk_df["TotalRiskScore"].mean() if not crew_risk_df.empty else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Claims", f"{int(total_claims)}")
    c2.metric("Total Claim Cost", f"${total_claim_cost:,.0f}")
    c3.metric("Utility Strikes", f"{int(total_strikes)}")
    c4.metric("Vehicle Accidents", f"{int(total_vehicle_accidents)}")
    c5.metric("Avg Crew Risk Score", f"{avg_risk_score:,.1f}")

    st.subheader("Crew Risk Summary")
    if not crew_risk_df.empty:
        show_cols = [
            "Crew", "HoursWorked", "OperationalScore", "FleetScore",
            "ClaimScore", "TotalRiskScore", "RiskLevel"
        ]
        st.dataframe(crew_risk_df[show_cols], use_container_width=True)
    else:
        st.info("No crew risk data available.")

# ============================================================
# PAGE: CREW RISK RANKING
# ============================================================
elif page == "Crew Risk Ranking":
    st.subheader("Crew Risk Ranking")

    if crew_risk_df.empty:
        st.info("No crew risk data available.")
    else:
        fig = px.bar(
            crew_risk_df,
            x="Crew",
            y="TotalRiskScore",
            color="RiskLevel",
            title="Total Risk Score by Crew",
            color_discrete_map={
                "Low": "green",
                "Moderate": "gold",
                "High": "orange",
                "Severe": "red",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            crew_risk_df[
                [
                    "Crew",
                    "HoursWorked",
                    "MilesDriven",
                    "OperationalPoints",
                    "FleetPoints",
                    "ClaimPoints",
                    "OperationalScore",
                    "FleetScore",
                    "ClaimScore",
                    "TotalRiskScore",
                    "RiskLevel",
                ]
            ],
            use_container_width=True,
        )

# ============================================================
# PAGE: CLAIMS ANALYTICS
# ============================================================
elif page == "Claims Analytics":
    st.subheader("Claims Analytics")

    if loss_df.empty:
        st.info("No claim data available.")
    else:
        left, right = st.columns(2)

        with left:
            if "ClaimType" in loss_df.columns:
                claim_type_counts = loss_df.groupby("ClaimType").size().reset_index(name="Count")
                fig = px.pie(claim_type_counts, names="ClaimType", values="Count", title="Claims by Type")
                st.plotly_chart(fig, use_container_width=True)

        with right:
            if "Cause" in loss_df.columns:
                cause_counts = (
                    loss_df.groupby("Cause").size().reset_index(name="Count").sort_values("Count", ascending=False)
                )
                fig = px.bar(cause_counts, x="Cause", y="Count", title="Claims by Cause")
                st.plotly_chart(fig, use_container_width=True)

        if {"IncidentDate", "ReportDate"}.issubset(loss_df.columns):
            lag_df = loss_df.copy()
            lag_df["LagDays"] = (lag_df["ReportDate"] - lag_df["IncidentDate"]).dt.days
            lag_df = lag_df.dropna(subset=["LagDays"])

            if not lag_df.empty:
                st.subheader("Claim Reporting Lag")
                st.metric("Average Lag Days", f"{lag_df['LagDays'].mean():.1f}")
                fig = px.histogram(lag_df, x="LagDays", nbins=15, title="Claim Lag Distribution")
                st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE: FLEET RISK
# ============================================================
elif page == "Fleet Risk":
    st.subheader("Fleet Risk")

    if fleet_df.empty:
        st.info("No fleet data available.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Miles Driven", f"{fleet_df['MilesDriven'].sum():,.0f}")
        c2.metric("Speeding Events", f"{fleet_df['SpeedingEvents'].sum():,.0f}")
        c3.metric("Harsh Brake Events", f"{fleet_df['HarshBrakeEvents'].sum():,.0f}")
        c4.metric("Vehicle Accidents", f"{fleet_df['VehicleAccidents'].sum():,.0f}")

        by_driver = fleet_df.groupby("Driver").agg(
            MilesDriven=("MilesDriven", "sum"),
            SpeedingEvents=("SpeedingEvents", "sum"),
            HarshBrakeEvents=("HarshBrakeEvents", "sum"),
            HarshAccelEvents=("HarshAccelEvents", "sum"),
            VehicleAccidents=("VehicleAccidents", "sum"),
        ).reset_index()

        fig = px.bar(by_driver, x="Driver", y="SpeedingEvents", title="Speeding Events by Driver")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(by_driver, use_container_width=True)

# ============================================================
# PAGE: UTILITY STRIKE TRACKER
# ============================================================
elif page == "Utility Strike Tracker":
    st.subheader("Utility Strike Tracker")

    if utility_df.empty:
        st.info("No utility strike data available.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Strikes", f"{len(utility_df):,}")
        c2.metric("Total Repair Cost", f"${utility_df['RepairCost'].sum():,.0f}")
        c3.metric("Average Repair Cost", f"${utility_df['RepairCost'].mean():,.0f}")

        left, right = st.columns(2)

        with left:
            by_type = utility_df.groupby("UtilityType").size().reset_index(name="Count")
            fig = px.bar(by_type, x="UtilityType", y="Count", title="Strikes by Utility Type")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            by_contractor = utility_df.groupby("HiringContractor").agg(
                Count=("StrikeID", "count"),
                RepairCost=("RepairCost", "sum"),
            ).reset_index()
            fig = px.bar(by_contractor, x="HiringContractor", y="RepairCost", title="Repair Cost by Hiring Contractor")
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(utility_df, use_container_width=True)

# ============================================================
# PAGE: DATA HEALTH
# ============================================================
elif page == "Data Health":
    st.subheader("Data Health Check")

    health = pd.DataFrame(
        [
            {"Dataset": "Loss Runs", "Rows": len(loss_df), "Columns": ", ".join(loss_df.columns)},
            {"Dataset": "Payroll Hours", "Rows": len(payroll_df), "Columns": ", ".join(payroll_df.columns)},
            {"Dataset": "Fleet Events", "Rows": len(fleet_df), "Columns": ", ".join(fleet_df.columns)},
            {"Dataset": "Incident Reports", "Rows": len(incident_df), "Columns": ", ".join(incident_df.columns)},
            {"Dataset": "Utility Strikes", "Rows": len(utility_df), "Columns": ", ".join(utility_df.columns)},
        ]
    )

    st.dataframe(health, use_container_width=True)

    st.markdown("### Expected File Names")
    st.code(
        """loss_runs.csv or loss_runs_sample.csv
payroll_hours.csv or payroll_hours_sample.csv
fleet_events.csv or fleet_events_sample.csv
incident_reports.csv or incident_reports_sample.csv
utility_strikes.csv or utility_strikes_sample.csv"""
    )

    st.markdown("### Expected Column Headers")

    st.markdown("**loss_runs**")
    st.code(",".join(LOSS_COLS))

    st.markdown("**payroll_hours**")
    st.code(",".join(PAYROLL_COLS))

    st.markdown("**fleet_events**")
    st.code(",".join(FLEET_COLS))

    st.markdown("**incident_reports**")
    st.code(",".join(INCIDENT_COLS))

    st.markdown("**utility_strikes**")
    st.code(",".join(UTILITY_COLS))
