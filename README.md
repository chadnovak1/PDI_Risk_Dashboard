# PDI Risk Dashboard

Internal Streamlit dashboard for tracking construction risk across claims, fleet safety, utility strikes, and incident reporting.

## Features

- **Executive Overview** – KPIs and crew risk summary
- **Crew Risk Ranking** – Comprehensive risk scores by crew
- **Claims Analytics** – TRIR, DART, frequency, and severity metrics
- **Fleet Risk** – Driver safety and vehicle accident tracking
- **Utility Strike Tracker** – Strike incidents and repair costs
- **Data Health Check** – Data validation and column verification
- **Data Upload** – Import updated CSV files with templates

## Project Structure

```
.
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── data/                     # Local CSV data files (user uploads)
├── sample_data/              # Sample CSV files for testing
└── venv/                     # Python virtual environment
```

## Quick Start

### 1. Initial Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
# Make sure venv is activated
source venv/bin/activate

# Start Streamlit
streamlit run app.py
```

The app will be available at `http://localhost:8503`

### 3. Load Data

- **Sample data** is provided in `sample_data/` for testing
- **User data** should be placed in `data/` directory or uploaded via the "Upload Data" page
- Use the "Data Health" page to validate your data structure

## Data Requirements

### File Names

- `loss_runs.csv` – Workers compensation & liability claims
- `payroll_hours.csv` – Payroll and hours worked
- `fleet_events.csv` – Vehicle telematics and incidents
- `incident_reports.csv` – Tool/equipment incidents and near misses
- `utility_strikes.csv` – Utility strike and locate incidents

### Data Upload

- Use the "Upload Data" page in the dashboard
- Download column templates to ensure proper format
- All files are saved to `/data` directory

## Configuration

Streamlit settings are in `.streamlit/config.toml`:
- Max upload size: 200 MB
- Theme: Light mode with PDI blue primary color
- Server settings optimized for dashboard use

## Development

To modify the dashboard:
1. Edit `app.py` with new features/pages
2. Restart the Streamlit server to see changes
3. Run `streamlit cache clear` if data isn't updating

## Troubleshooting

### "streamlit: command not found"
Ensure the virtual environment is activated:
```bash
source venv/bin/activate
```

### Data not loading
1. Check the "Data Health" page to verify files exist
2. Ensure CSV files have the correct column headers
3. Files should be in `data/` or uploaded via the dashboard

### Deprecation warnings
The dashboard uses current Streamlit APIs. Warnings are suppressed in the config.
