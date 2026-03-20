import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path


try:
    from fpdf import FPDF
    fpdf_available = True
except ImportError:
    FPDF = None
    fpdf_available = False


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
    # Convert columns to numeric, handling common CSV formatting (commas, dollar signs).
    for col in cols:
        if col in df.columns:
            # Convert to string first to ensure .str methods are available
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[,$]", "", regex=True)
                .replace({"nan": None, "None": None})
            )
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


def validate_upload(uploaded_file, expected_cols: list[str]) -> tuple[bool, str, pd.DataFrame]:
    """Validate and load an uploaded CSV file."""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check if all expected columns are present
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            return False, f"Missing columns: {', '.join(missing_cols)}", pd.DataFrame()
        
        return True, "✓ File validated successfully", df
    except Exception as e:
        return False, f"Error reading file: {str(e)}", pd.DataFrame()


def save_uploaded_file(file_path: Path, df: pd.DataFrame) -> bool:
    """Save uploaded dataframe to file."""
    try:
        df.to_csv(file_path, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return False


def create_template_df(columns: list[str], num_rows: int = 2) -> pd.DataFrame:
    """Create a template dataframe with specified columns and empty rows."""
    return pd.DataFrame({col: [""] * num_rows for col in columns})


def generate_csv_download(df: pd.DataFrame, filename: str) -> bytes:
    """Generate CSV bytes from dataframe for download."""
    return df.to_csv(index=False).encode('utf-8')


def generate_pdf_from_summary(summary_data: dict) -> bytes:
    """Generate a simple PDF bytes object from summary dictionary."""
    if not fpdf_available:
        raise RuntimeError("fpdf package is not available. Install via pip install fpdf")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Injury Triage Summary", ln=True, align="C")
    pdf.ln(4)

    pdf.set_font("Arial", size=11)
    for field, value in summary_data.items():
        # Clean the value to remove/replace problematic Unicode characters
        clean_value = str(value).replace("\n", " ")
        # Replace common Unicode characters that cause issues
        clean_value = clean_value.replace("–", "-").replace("—", "-").replace("'", "'").replace('"', '"')
        # Remove any remaining non-ASCII characters
        clean_value = ''.join(c for c in clean_value if ord(c) < 128)
        line = f"{field}: {clean_value}"
        pdf.multi_cell(0, 7, line)

    try:
        out = pdf.output(dest="S")
        if isinstance(out, str):
            out = out.encode("latin-1")
        return out
    except UnicodeEncodeError:
        # Fallback: create a simpler PDF without problematic characters
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=10)
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Injury Triage Summary", ln=True, align="C")
        pdf.ln(4)

        pdf.set_font("Arial", size=11)
        for field, value in summary_data.items():
            # More aggressive cleaning for fallback
            clean_value = str(value).replace("\n", " ")
            clean_value = ''.join(c for c in clean_value if ord(c) < 128 and c.isprintable())
            line = f"{field}: {clean_value}"
            pdf.multi_cell(0, 7, line)

        out = pdf.output(dest="S")
        if isinstance(out, str):
            out = out.encode("latin-1")
        return out


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
    "Reserved", "TotalIncurred", "WC Claim Type", "Status"
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
loss_df = to_numeric_safe(loss_df, ["Paid", "Reserved", "TotalIncurred"])
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
elif "WC Claim Type" in loss_df.columns:
    # Derive recordable from WC Claim Type (counts toward TRIR)
    # - Anything mentioning "medical" or "lost" is treated as recordable.
    # - "Not Applicable" / "N/A" is treated as non-recordable.
    wc_type = loss_df["WC Claim Type"].astype(str).str.strip().str.lower()
    loss_df["Recordable"] = (~wc_type.isin(["not applicable", "n/a", ""])) & (
        wc_type.str.contains("medical") | wc_type.str.contains("lost")
    )
    loss_df["Recordable"] = loss_df["Recordable"].astype(int)

if "LostTime" in loss_df.columns:
    loss_df["LostTime"] = yes_no_to_int(loss_df["LostTime"])
elif "WC Claim Type" in loss_df.columns:
    # Derive lost-time from WC Claim Type (counts toward DART)
    wc_type = loss_df["WC Claim Type"].astype(str).str.strip().str.lower()
    loss_df["LostTime"] = (
        wc_type.str.contains("lost")
    ).astype(int)

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
# CLAIMS ANALYTICS HELPERS
# ============================================================
def calculate_claims_metrics():
    """Calculate TRIR, DART, Incident Frequency, and Severity metrics by crew."""
    if loss_df.empty or payroll_df.empty:
        return pd.DataFrame()

    # Get total hours by crew
    crew_hours = payroll_df.groupby("Crew", dropna=False)["HoursWorked"].sum().reset_index()
    crew_hours.columns = ["Crew", "TotalHours"]

    # Get claim metrics by crew
    claims_agg = loss_df.groupby("Crew").agg(
        TotalClaims=("ClaimNumber", "count"),
        TotalClaimCost=("TotalIncurred", "sum"),
    ).reset_index()

    # Add optional columns if present
    if "Recordable" in loss_df.columns:
        recordable = loss_df.groupby("Crew")["Recordable"].sum().reset_index(name="RecordableClaims")
        claims_agg = claims_agg.merge(recordable, on="Crew", how="left")
    else:
        claims_agg["RecordableClaims"] = 0

    if "LostTime" in loss_df.columns:
        lost_time = loss_df.groupby("Crew")["LostTime"].sum().reset_index(name="LostTimeClaims")
        claims_agg = claims_agg.merge(lost_time, on="Crew", how="left")
    else:
        claims_agg["LostTimeClaims"] = 0

    # Merge
    metrics_df = crew_hours.merge(claims_agg, on="Crew", how="left").fillna(0)

    # Calculate rates (using 200,000 base hours per OSHA standard)
    metrics_df["TRIR"] = np.where(
        metrics_df["TotalHours"] > 0,
        (metrics_df["RecordableClaims"] * 200000) / metrics_df["TotalHours"],
        0,
    )

    metrics_df["DART"] = np.where(
        metrics_df["TotalHours"] > 0,
        (metrics_df["LostTimeClaims"] * 200000) / metrics_df["TotalHours"],
        0,
    )

    # Get incident frequency by crew
    incident_counts = incident_df.groupby("Crew").size().reset_index(name="IncidentCount")
    metrics_df = metrics_df.merge(incident_counts, on="Crew", how="left").fillna({
        "IncidentCount": 0
    })

    metrics_df["IncidentFrequency"] = np.where(
        metrics_df["TotalHours"] > 0,
        (metrics_df["IncidentCount"] * 200000) / metrics_df["TotalHours"],
        0,
    )

    # Calculate severity (average cost per claim)
    metrics_df["Severity"] = np.where(
        metrics_df["TotalClaims"] > 0,
        metrics_df["TotalClaimCost"] / metrics_df["TotalClaims"],
        0,
    )

    return metrics_df.sort_values("TRIR", ascending=False)


def get_metrics_by_period():
    """Calculate TRIR and DART by month for trend analysis."""
    if loss_df.empty or payroll_df.empty:
        return pd.DataFrame()

    # Monthly payroll totals
    monthly_hours = payroll_df.groupby(["Year", "Month"])["HoursWorked"].sum().reset_index()
    monthly_hours.columns = ["Year", "Month", "TotalHours"]

    # Monthly claim metrics
    loss_df_copy = loss_df.copy()
    loss_df_copy["Year"] = loss_df_copy["IncidentDate"].dt.year
    loss_df_copy["Month"] = loss_df_copy["IncidentDate"].dt.month

    # Monthly claim metrics (Recordable/LostTime are optional)
    agg_map = {}
    if "Recordable" in loss_df_copy.columns:
        agg_map["RecordableClaims"] = ("Recordable", "sum")
    if "LostTime" in loss_df_copy.columns:
        agg_map["LostTimeClaims"] = ("LostTime", "sum")

    monthly_claims = loss_df_copy.groupby(["Year", "Month"]).agg(**agg_map).reset_index()

    if "RecordableClaims" not in monthly_claims.columns:
        monthly_claims["RecordableClaims"] = 0
    if "LostTimeClaims" not in monthly_claims.columns:
        monthly_claims["LostTimeClaims"] = 0

    # Merge
    trends_df = monthly_hours.merge(monthly_claims, on=["Year", "Month"], how="left").fillna(0)
    trends_df["Date"] = pd.to_datetime(trends_df[["Year", "Month"]].assign(DAY=1))

    # Calculate rates
    trends_df["TRIR"] = np.where(
        trends_df["TotalHours"] > 0,
        (trends_df["RecordableClaims"] * 200000) / trends_df["TotalHours"],
        0,
    )

    trends_df["DART"] = np.where(
        trends_df["TotalHours"] > 0,
        (trends_df["LostTimeClaims"] * 200000) / trends_df["TotalHours"],
        0,
    )

    return trends_df.sort_values("Date")

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
    ).reset_index()

    if "Recordable" in loss_df.columns:
        recordable = loss_df.groupby("Crew")["Recordable"].sum().reset_index(name="RecordableClaims")
        claims_agg = claims_agg.merge(recordable, on="Crew", how="left")
    else:
        claims_agg["RecordableClaims"] = 0

    if "LostTime" in loss_df.columns:
        lost_time = loss_df.groupby("Crew")["LostTime"].sum().reset_index(name="LostTimeClaims")
        claims_agg = claims_agg.merge(lost_time, on="Crew", how="left")
    else:
        claims_agg["LostTimeClaims"] = 0

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


def render_injury_triage_tool():
    st.subheader("Employee Injury Triage Tool")
    st.caption("Use this decision tree to determine whether clinic referral is required.")

    # Basic info
    st.markdown("### Basic Information")
    c1, c2 = st.columns(2)
    employee_name = c1.text_input("Employee Name")
    supervisor_name = c2.text_input("Supervisor Name")

    c3, c4, c5 = st.columns(3)
    injury_date = c3.date_input("Date of Injury")
    injury_time = c4.time_input("Time of Injury")
    location = c5.text_input("Job Site / Location")

    incident_description = st.text_area("Describe what happened")

    st.divider()

    # ============================================================
    # STEP 1 – RED FLAGS
    # ============================================================
    st.markdown("### STEP 1 – RED FLAGS")
    st.write(
        "If **ANY** red flag is present, stop and send the employee to the clinic immediately."
    )

    red_flags = st.multiselect(
        "Check any that apply:",
        [
            "Loss of consciousness",
            "Head injury",
            "Eye injury",
            "Severe bleeding / cut will not stop",
            "Suspected fracture / cannot bear weight",
            "Crush injury",
            "Severe pain (7/10 or greater)",
            "Chest pain / breathing issue",
            "Electrical contact",
            "Burn (more than minor)",
            "NONE OF THE ABOVE",
        ],
        key="red_flags",
    )

    has_red_flag = any(flag != "NONE OF THE ABOVE" for flag in red_flags)

    if has_red_flag:
        st.error("Red flag present: **Send employee to clinic immediately and notify Chad.**")
        red_flag_action = st.radio(
            "Confirm action:",
            ["Employee sent to clinic", "Employee will be sent immediately"],
            key="red_flag_action",
        )
    else:
        red_flag_action = None

    st.divider()

    # ============================================================
    # STEP 2 – WORK ABILITY
    # ============================================================
    work_ability = None
    work_ability_action = None
    if not has_red_flag:
        st.markdown("### STEP 2 – WORK ABILITY")
        work_ability = st.radio(
            "Can the employee safely perform their normal job duties right now?",
            [
                "Yes – employee can safely perform normal duties",
                "No – employee cannot safely perform normal duties",
            ],
            key="work_ability",
        )

        if work_ability.startswith("No"):
            st.error("Employee cannot safely perform normal duties: **Clinic referral is required.**")
            work_ability_action = st.radio(
                "Confirm action:",
                ["Employee sent to clinic", "Employee will be sent immediately"],
                key="work_ability_action",
            )

    st.divider()

    # ============================================================
    # STEP 3 – SYMPTOMS
    # ============================================================
    symptoms = []
    symptoms_action = None
    can_continue_to_symptoms = (
        not has_red_flag and work_ability and work_ability.startswith("Yes")
    )

    if can_continue_to_symptoms:
        st.markdown("### STEP 3 – SYMPTOMS")
        st.write("If **any** symptoms are present, clinic referral is required.")

        symptoms = st.multiselect(
            "Check any that apply:",
            [
                "Swelling",
                "Limited movement",
                "Increasing pain",
                "NONE OF THE ABOVE",
            ],
            key="symptoms",
        )

        has_symptoms = any(item != "NONE OF THE ABOVE" for item in symptoms)

        if has_symptoms:
            st.warning("Symptoms present: **Employee should be referred to the clinic.**")
            symptoms_action = st.radio(
                "Confirm action:",
                ["Employee sent to clinic", "Employee will be sent immediately"],
                key="symptoms_action",
            )
    else:
        has_symptoms = False

    st.divider()

    # ============================================================
    # STEP 4 – MECHANISM OF INJURY
    # ============================================================
    mechanism = []
    mechanism_action = None
    mechanism_pain_followup = None
    can_continue_to_mechanism = (
        not has_red_flag
        and work_ability
        and work_ability.startswith("Yes")
        and not has_symptoms
    )

    if can_continue_to_mechanism:
        st.markdown("### STEP 4 – MECHANISM OF INJURY")
        st.write(
            "These mechanisms increase the risk of delayed injury. "
            "Mechanism alone does not require clinic referral, but worsening pain or limited movement may."
        )

        mechanism = st.multiselect(
            "Check any that apply:",
            [
                "Heavy lifting",
                "Awkward twist or strain",
                "Fall (same level or elevated)",
                "Struck by object or equipment",
                "NONE OF THE ABOVE",
            ],
            key="mechanism",
        )

        has_mechanism_flag = any(item != "NONE OF THE ABOVE" for item in mechanism)

        if has_mechanism_flag:
            st.info(
                "A higher-risk mechanism is present. Evaluate the employee's current condition before deciding."
            )

            mechanism_pain_followup = st.radio(
                "Is the employee experiencing pain that is worsening or limiting movement?",
                ["No", "Yes"],
                key="mechanism_pain_followup",
            )

            if mechanism_pain_followup == "Yes":
                st.warning("Worsening pain or limited movement present: **Clinic referral is required.**")
                mechanism_action = st.radio(
                    "Confirm action:",
                    ["Employee sent to clinic", "Employee will be sent immediately"],
                    key="mechanism_action",
                )
            else:
                mechanism_action = st.radio(
                    "Based on the mechanism and current condition, what is the appropriate action?",
                    ["Continue with first aid and monitor", "Refer to clinic"],
                    key="mechanism_action_non_trigger",
                )
    else:
        has_mechanism_flag = False

    st.divider()

    # ============================================================
    # STEP 5 – FIRST AID
    # ============================================================
    first_aid_measures = []
    basic_first_aid_only = None
    follow_up = None
    next_day_return = None

    can_continue_to_first_aid = (
        not has_red_flag
        and work_ability
        and work_ability.startswith("Yes")
        and not has_symptoms
        and (
            not has_mechanism_flag
            or (mechanism_pain_followup == "No" and mechanism_action == "Continue with first aid and monitor")
        )
    )

    if can_continue_to_first_aid:
        st.markdown("### STEP 5 – FIRST AID")
        st.success("No red flags or symptom triggers requiring clinic referral were identified.")

        first_aid_measures = st.multiselect(
            "Select all first aid measures and supplies provided:",
            [
                "Rest",
                "Ice",
                "Heat",
                "Stretch",
                "Bandage / dressing",
                "Antiseptic / wound cleaning",
                "Ice pack (instant or reusable)",
                "Compression wrap",
                "Elevation",
                "None provided",
            ],
            key="first_aid_measures",
        )

        basic_first_aid_only = st.radio(
            "Was only basic first aid provided (no medical treatment beyond first aid)?",
            ["Yes", "No"],
            key="basic_first_aid_only",
        )

        follow_up = st.radio(
            "Has a next-day follow-up been scheduled?",
            ["Yes – follow-up scheduled", "No – follow-up not scheduled"],
            key="follow_up",
        )

        if follow_up.startswith("No"):
            st.warning("Follow-up is required for all first aid cases.")
            st.radio(
                "Confirm action:",
                ["Follow-up will be scheduled immediately"],
                key="follow_up_action",
            )

        next_day_return = st.radio(
            "If the employee cannot report to work the next day due to this injury, clinic evaluation is required. "
            "Based on current condition, do you expect the employee will be able to report to work the next day?",
            ["Yes", "No / Unsure"],
            key="next_day_return",
        )

        if next_day_return == "No / Unsure":
            st.warning("If the employee may not be able to report to work the next day, **clinic evaluation is required.**")
            st.radio(
                "Confirm action:",
                ["Employee sent to clinic", "Employee will be sent immediately"],
                key="next_day_return_action",
            )

    st.divider()

    # ============================================================
    # FINAL DETERMINATION
    # ============================================================
    st.markdown("### FINAL DETERMINATION")

    referred_due_to_mechanism = (
        has_mechanism_flag and (
            (mechanism_pain_followup == "Yes")
            or (mechanism_action == "Refer to clinic")
        )
    )

    referred_due_to_next_day = (
        can_continue_to_first_aid and next_day_return == "No / Unsure"
    )

    if has_red_flag:
        recommended_outcome = "Referred to Clinic"
        trigger_reason = "Red Flag"
    elif work_ability and work_ability.startswith("No"):
        recommended_outcome = "Referred to Clinic"
        trigger_reason = "Unable to Perform Normal Duties"
    elif has_symptoms:
        recommended_outcome = "Referred to Clinic"
        trigger_reason = "Symptoms Present"
    elif referred_due_to_mechanism:
        recommended_outcome = "Referred to Clinic"
        trigger_reason = "Mechanism + Current Condition"
    elif referred_due_to_next_day:
        recommended_outcome = "Referred to Clinic"
        trigger_reason = "May Not Be Able to Report to Work Next Day"
    elif can_continue_to_first_aid:
        recommended_outcome = "First Aid Only"
        trigger_reason = "No Escalation Trigger Met"
    else:
        recommended_outcome = "Incomplete"
        trigger_reason = "Decision Tree Not Completed"

    st.info(f"Recommended Outcome: **{recommended_outcome}**")
    st.write(f"Trigger Reason: **{trigger_reason}**")

    final_outcome = st.radio(
        "Select the final outcome:",
        ["Referred to Clinic", "First Aid Only"],
        index=0 if recommended_outcome == "Referred to Clinic" else 1,
        key="final_outcome",
    )

    notified = st.checkbox(
        "I confirm that I followed the injury triage process and have notified Chad.",
        key="notified_chad",
    )

    st.divider()

    # ============================================================
    # SUMMARY
    # ============================================================
    st.markdown("### Triage Summary")

    summary_data = {
        "Employee Name": employee_name,
        "Supervisor Name": supervisor_name,
        "Job Site / Location": location,
        "Incident Description": incident_description,
        "Red Flags": ", ".join(red_flags) if red_flags else "",
        "Work Ability": work_ability if work_ability else "",
        "Symptoms": ", ".join(symptoms) if symptoms else "",
        "Mechanism": ", ".join(mechanism) if mechanism else "",
        "Mechanism Pain / Limiting Movement": mechanism_pain_followup if mechanism_pain_followup else "",
        "Mechanism Decision": mechanism_action if mechanism_action else "",
        "First Aid Measures": ", ".join(first_aid_measures) if first_aid_measures else "",
        "Basic First Aid Only": basic_first_aid_only if basic_first_aid_only else "",
        "Next-Day Follow-Up": follow_up if follow_up else "",
        "Expected Next-Day Return": next_day_return if next_day_return else "",
        "Recommended Outcome": recommended_outcome,
        "Trigger Reason": trigger_reason,
        "Final Outcome": final_outcome,
        "Notified Chad": "Yes" if notified else "No",
    }

    summary_df = pd.DataFrame(summary_data.items(), columns=["Field", "Value"])
    st.dataframe(summary_df, use_container_width=True)

    csv = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Triage Summary CSV",
        data=csv,
        file_name="injury_triage_summary.csv",
        mime="text/csv",
    )


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
        "Injury Triage Tool",
        "Data Health",
        "Upload Data",
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
        st.dataframe(crew_risk_df[show_cols], width='stretch')
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
        st.plotly_chart(fig, width='stretch')

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
            width='stretch',
        )

# ============================================================
# PAGE: CLAIMS ANALYTICS
# ============================================================
elif page == "Claims Analytics":
    st.subheader("Claims Analytics")

    if loss_df.empty:
        st.info("No claim data available.")
    else:
        # Overview metrics
        claims_metrics = calculate_claims_metrics()
        
        if not claims_metrics.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg TRIR", f"{claims_metrics['TRIR'].mean():.2f}")
            c2.metric("Avg DART", f"{claims_metrics['DART'].mean():.2f}")
            c3.metric("Avg Incident Freq", f"{claims_metrics['IncidentFrequency'].mean():.2f}")
            c4.metric("Avg Severity ($)", f"${claims_metrics['Severity'].mean():,.0f}")
        
        st.markdown("---")
        st.subheader("Key Safety Metrics by Crew")
        
        if not claims_metrics.empty:
            left, right = st.columns(2)
            
            with left:
                fig_trir = px.bar(
                    claims_metrics,
                    x="Crew",
                    y="TRIR",
                    title="TRIR (Total Recordable Incident Rate) by Crew",
                    color="TRIR",
                    color_continuous_scale="Reds",
                    labels={"TRIR": "TRIR (per 200k hrs)"}
                )
                st.plotly_chart(fig_trir, width='stretch')
            
            with right:
                fig_dart = px.bar(
                    claims_metrics,
                    x="Crew",
                    y="DART",
                    title="DART (Days Away/Restricted/Transferred) by Crew",
                    color="DART",
                    color_continuous_scale="Oranges",
                    labels={"DART": "DART (per 200k hrs)"}
                )
                st.plotly_chart(fig_dart, width='stretch')
        
        st.markdown("---")
        
        left, right = st.columns(2)
        
        with left:
            if not claims_metrics.empty:
                fig_freq = px.bar(
                    claims_metrics,
                    x="Crew",
                    y="IncidentFrequency",
                    title="Incident Frequency by Crew",
                    color="IncidentFrequency",
                    color_continuous_scale="Blues",
                    labels={"IncidentFrequency": "Frequency (per 200k hrs)"}
                )
                st.plotly_chart(fig_freq, width='stretch')
        
        with right:
            if not claims_metrics.empty:
                fig_severity = px.bar(
                    claims_metrics,
                    x="Crew",
                    y="Severity",
                    title="Average Claim Severity by Crew",
                    color="Severity",
                    color_continuous_scale="Purples",
                    labels={"Severity": "Avg Cost per Claim ($)"}
                )
                st.plotly_chart(fig_severity, width='stretch')
        
        st.markdown("---")
        st.subheader("Trends Over Time")
        
        trends_df = get_metrics_by_period()
        if not trends_df.empty:
            left, right = st.columns(2)
            
            with left:
                fig_trir_trend = px.line(
                    trends_df,
                    x="Date",
                    y="TRIR",
                    title="TRIR Trend",
                    markers=True,
                    labels={"TRIR": "TRIR (per 200k hrs)", "Date": "Month"}
                )
                st.plotly_chart(fig_trir_trend, width='stretch')
            
            with right:
                fig_dart_trend = px.line(
                    trends_df,
                    x="Date",
                    y="DART",
                    title="DART Trend",
                    markers=True,
                    labels={"DART": "DART (per 200k hrs)", "Date": "Month"}
                )
                st.plotly_chart(fig_dart_trend, width='stretch')
        
        st.markdown("---")
        st.subheader("Claims Distribution")
        
        left, right = st.columns(2)

        with left:
            if "ClaimType" in loss_df.columns:
                claim_type_counts = loss_df.groupby("ClaimType").size().reset_index(name="Count")
                fig = px.pie(claim_type_counts, names="ClaimType", values="Count", title="Claims by Type")
                st.plotly_chart(fig, width='stretch')

        with right:
            if "Cause" in loss_df.columns:
                cause_counts = (
                    loss_df.groupby("Cause").size().reset_index(name="Count").sort_values("Count", ascending=False)
                )
                fig = px.bar(cause_counts, x="Cause", y="Count", title="Claims by Cause", height=400)
                st.plotly_chart(fig, width='stretch')

        if {"IncidentDate", "ReportDate"}.issubset(loss_df.columns):
            lag_df = loss_df.copy()
            lag_df["LagDays"] = (lag_df["ReportDate"] - lag_df["IncidentDate"]).dt.days
            lag_df = lag_df.dropna(subset=["LagDays"])

            if not lag_df.empty:
                st.subheader("Claim Reporting Lag")
                st.metric("Average Lag Days", f"{lag_df['LagDays'].mean():.1f}")
                fig = px.histogram(lag_df, x="LagDays", nbins=15, title="Claim Lag Distribution")
                st.plotly_chart(fig, width='stretch')

        st.markdown("---")
        st.subheader("Detailed Metrics by Crew")
        if not claims_metrics.empty:
            display_cols = ["Crew", "TotalHours", "TotalClaims", "RecordableClaims", 
                          "LostTimeClaims", "TRIR", "DART", "IncidentFrequency", "Severity"]
            st.dataframe(claims_metrics[display_cols], width='stretch')

# ============================================================
# PAGE: FLEET RISK
# ============================================================
# ============================================================
# PAGE: INJURY TRIAGE TOOL
elif page == "Injury Triage Tool":
    render_injury_triage_tool()

# ============================================================
# PAGE: FLEET RISK
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
        st.plotly_chart(fig, width='stretch')

        st.dataframe(by_driver, width='stretch')

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
            st.plotly_chart(fig, width='stretch')

        with right:
            by_contractor = utility_df.groupby("HiringContractor").agg(
                Count=("StrikeID", "count"),
                RepairCost=("RepairCost", "sum"),
            ).reset_index()
            fig = px.bar(by_contractor, x="HiringContractor", y="RepairCost", title="Repair Cost by Hiring Contractor")
            st.plotly_chart(fig, width='stretch')

        st.dataframe(utility_df, width='stretch')

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

    st.dataframe(health, width='stretch')

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

# ============================================================
# PAGE: UPLOAD DATA
# ============================================================
elif page == "Upload Data":
    st.subheader("Upload Data Files")
    st.markdown(
        """
        Upload updated CSV files to replace the current data. Files are saved to the `/data` directory.
        Download templates below to ensure your data has all required columns.
        """
    )

    st.markdown("---")
    st.subheader("📥 Download Templates")
    st.markdown("Click the buttons below to download CSV templates. Fill them out with your data and upload them back.")
    
    template_cols = st.columns(5)
    
    with template_cols[0]:
        template_loss = create_template_df(LOSS_COLS, num_rows=2)
        st.download_button(
            label="📄 Loss Runs",
            data=generate_csv_download(template_loss, "loss_runs_template.csv"),
            file_name="loss_runs_template.csv",
            mime="text/csv",
            key="dl_loss_template"
        )
    
    with template_cols[1]:
        template_payroll = create_template_df(PAYROLL_COLS, num_rows=2)
        st.download_button(
            label="📄 Payroll Hours",
            data=generate_csv_download(template_payroll, "payroll_hours_template.csv"),
            file_name="payroll_hours_template.csv",
            mime="text/csv",
            key="dl_payroll_template"
        )
    
    with template_cols[2]:
        template_fleet = create_template_df(FLEET_COLS, num_rows=2)
        st.download_button(
            label="📄 Fleet Events",
            data=generate_csv_download(template_fleet, "fleet_events_template.csv"),
            file_name="fleet_events_template.csv",
            mime="text/csv",
            key="dl_fleet_template"
        )
    
    with template_cols[3]:
        template_incident = create_template_df(INCIDENT_COLS, num_rows=2)
        st.download_button(
            label="📄 Incidents",
            data=generate_csv_download(template_incident, "incident_reports_template.csv"),
            file_name="incident_reports_template.csv",
            mime="text/csv",
            key="dl_incident_template"
        )
    
    with template_cols[4]:
        template_utility = create_template_df(UTILITY_COLS, num_rows=2)
        st.download_button(
            label="📄 Utility Strikes",
            data=generate_csv_download(template_utility, "utility_strikes_template.csv"),
            file_name="utility_strikes_template.csv",
            mime="text/csv",
            key="dl_utility_template"
        )

    st.markdown("---")
    st.subheader("📋 Expected Column Reference")
    
    with st.expander("Loss Runs Columns", expanded=False):
        st.info(
            """
            **Note:**
            - `IncidentDate` and `ReportDate` should be in MM/DD/YYYY format.
            - `Recordable` and `LostTime` may be derived from `WC Claim Type` if missing.
            - For TRIR/DART, `WC Claim Type` values **Medical Only**, **Lost Time**, **Became Lost Time**, and **Became Medical Only** count as **Yes / 1**.
            - `Not Applicable` or `N/A` counts as **No / 0**.
            """
        )
        loss_ref = pd.DataFrame({"Column": LOSS_COLS, "Example": [
            "WC001", "Workers Comp", "3/10/2026", "3/11/2026", "Pot Hole Crew",
            "John Smith", "John Doe", "Hand", "Cut from hand tool", "1200",
            "3000", "4200", "Medical Only", "Open"
        ]})
        st.dataframe(loss_ref, width='stretch', hide_index=True)
    
    with st.expander("Payroll Hours Columns", expanded=False):
        st.info("**Note:** Year and Month should be numeric (e.g., 2026, 3)")
        payroll_ref = pd.DataFrame({"Column": PAYROLL_COLS, "Example": [
            "2026", "3", "Pot Hole Crew", "3900", "5", "87000"
        ]})
        st.dataframe(payroll_ref, width='stretch', hide_index=True)
    
    with st.expander("Fleet Events Columns", expanded=False):
        st.info("**Note:** Date should be in MM/DD/YYYY format. All event counts should be numeric.")
        fleet_ref = pd.DataFrame({"Column": FLEET_COLS, "Example": [
            "3/10/2026", "F350-12", "Driver B", "Locator", "Pot Hole Crew",
            "175", "5.8", "2", "0", "1", "0", "0", "0", "0.8"
        ]})
        st.dataframe(fleet_ref, width='stretch', hide_index=True)
    
    with st.expander("Incident Reports Columns", expanded=False):
        st.info("**Note:** Date should be in MM/DD/YYYY format. CitationIssued and SafetyViolation should be: Yes/No, True/False, or 1/0")
        incident_ref = pd.DataFrame({"Column": INCIDENT_COLS, "Example": [
            "AA1", "3/9/2026", "12:30", "Vac Truck", "John Smith", "Driver A",
            "Auto Accident", "Vac Truck", "Yes", "Yes", "Stream7", "APS"
        ]})
        st.dataframe(incident_ref, width='stretch', hide_index=True)
    
    with st.expander("Utility Strikes Columns", expanded=False):
        st.info("**Note:** Date should be in MM/DD/YYYY format. Located, ToleranceZone, and CitationIssued should be: Yes/No, True/False, or 1/0. RepairCost should be numeric.")
        utility_ref = pd.DataFrame({"Column": UTILITY_COLS, "Example": [
            "GL1", "3/11/2026", "1:30", "Equipment Operator", "Operator", "John Smith",
            "Excavator", "Fiber", "Yes", "Yes", "4500", "Yes", "Frye Rd", "SRP"
        ]})
        st.dataframe(utility_ref, width='stretch', hide_index=True)

    st.markdown("---")
    st.subheader("⬆️ Upload Your Data")

    # Loss Runs Upload
    st.markdown("### Loss Runs (Workers Comp & Liability Claims)")
    loss_upload = st.file_uploader(
        "Upload loss_runs.csv",
        type="csv",
        key="loss_uploader",
        accept_multiple_files=False,
    )
    if loss_upload:
        is_valid, msg, df = validate_upload(loss_upload, LOSS_COLS)
        if is_valid:
            st.success(msg)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"✓ File contains {len(df)} rows and {len(df.columns)} columns")
            with col2:
                if st.button("Save Loss Runs", key="save_loss"):
                    file_path = DATA_DIR / "loss_runs.csv"
                    if save_uploaded_file(file_path, df):
                        st.success(f"✓ Saved! Refresh to see changes.")
        else:
            st.error(msg)

    st.markdown("---")

    # Payroll Hours Upload
    st.markdown("### Payroll Hours")
    payroll_upload = st.file_uploader(
        "Upload payroll_hours.csv",
        type="csv",
        key="payroll_uploader",
        accept_multiple_files=False,
    )
    if payroll_upload:
        is_valid, msg, df = validate_upload(payroll_upload, PAYROLL_COLS)
        if is_valid:
            st.success(msg)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"✓ File contains {len(df)} rows and {len(df.columns)} columns")
            with col2:
                if st.button("Save Payroll Hours", key="save_payroll"):
                    file_path = DATA_DIR / "payroll_hours.csv"
                    if save_uploaded_file(file_path, df):
                        st.success(f"✓ Saved! Refresh to see changes.")
        else:
            st.error(msg)

    st.markdown("---")

    # Fleet Events Upload
    st.markdown("### Fleet Events")
    fleet_upload = st.file_uploader(
        "Upload fleet_events.csv",
        type="csv",
        key="fleet_uploader",
        accept_multiple_files=False,
    )
    if fleet_upload:
        is_valid, msg, df = validate_upload(fleet_upload, FLEET_COLS)
        if is_valid:
            st.success(msg)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"✓ File contains {len(df)} rows and {len(df.columns)} columns")
            with col2:
                if st.button("Save Fleet Events", key="save_fleet"):
                    file_path = DATA_DIR / "fleet_events.csv"
                    if save_uploaded_file(file_path, df):
                        st.success(f"✓ Saved! Refresh to see changes.")
        else:
            st.error(msg)

    st.markdown("---")

    # Incident Reports Upload
    st.markdown("### Incident Reports")
    incident_upload = st.file_uploader(
        "Upload incident_reports.csv",
        type="csv",
        key="incident_uploader",
        accept_multiple_files=False,
    )
    if incident_upload:
        is_valid, msg, df = validate_upload(incident_upload, INCIDENT_COLS)
        if is_valid:
            st.success(msg)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"✓ File contains {len(df)} rows and {len(df.columns)} columns")
            with col2:
                if st.button("Save Incident Reports", key="save_incident"):
                    file_path = DATA_DIR / "incident_reports.csv"
                    if save_uploaded_file(file_path, df):
                        st.success(f"✓ Saved! Refresh to see changes.")
        else:
            st.error(msg)

    st.markdown("---")

    # Utility Strikes Upload
    st.markdown("### Utility Strikes")
    utility_upload = st.file_uploader(
        "Upload utility_strikes.csv",
        type="csv",
        key="utility_uploader",
        accept_multiple_files=False,
    )
    if utility_upload:
        is_valid, msg, df = validate_upload(utility_upload, UTILITY_COLS)
        if is_valid:
            st.success(msg)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"✓ File contains {len(df)} rows and {len(df.columns)} columns")
            with col2:
                if st.button("Save Utility Strikes", key="save_utility"):
                    file_path = DATA_DIR / "utility_strikes.csv"
                    if save_uploaded_file(file_path, df):
                        st.success(f"✓ Saved! Refresh to see changes.")
        else:
            st.error(msg)
