# Deployment Guide

This guide covers various deployment options for the PDI Risk Dashboard.

## Local Development

For development on your local machine:

```bash
# One-time setup
./setup.sh

# Start the dashboard
./start.sh
```

## Deployment Options

### Option 1: Streamlit Cloud (Recommended for Demos)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Create new app and select your repository
5. Specify `app.py` as the entry point

**Note:** Streamlit Cloud includes free tier with limitations. Ensure data files are in `sample_data/` for demos.

### Option 2: Docker Container

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8503

CMD ["streamlit", "run", "app.py", "--server.port=8503", "--server.address=0.0.0.0"]
```

Build and run:

```bash
docker build -t pdi-risk-dashboard .
docker run -p 8503:8503 -v $(pwd)/data:/app/data pdi-risk-dashboard
```

### Option 3: AWS EC2

1. Launch EC2 instance (Ubuntu 22.04)
2. Connect via SSH
3. Install Python3 and git:
   ```bash
   sudo apt update && sudo apt install -y python3 python3-pip git
   ```
4. Clone repository and setup:
   ```bash
   git clone https://github.com/chadnovak1/PDI_Risk_Dashboard.git
   cd PDI_Risk_Dashboard
   ./setup.sh
   ```
5. Run with systemd:
   ```bash
   sudo nano /etc/systemd/system/pdi-dashboard.service
   ```
   Add:
   ```ini
   [Unit]
   Description=PDI Risk Dashboard
   After=network.target

   [Service]
   Type=simple
   User=ubuntu
   WorkingDirectory=/home/ubuntu/PDI_Risk_Dashboard
   ExecStart=/home/ubuntu/PDI_Risk_Dashboard/venv/bin/streamlit run app.py --server.port=8503
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```
   Enable and start:
   ```bash
   sudo systemctl enable pdi-dashboard
   sudo systemctl start pdi-dashboard
   ```

### Option 4: Heroku

1. Create `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Create `runtime.txt`:
   ```
   python-3.11.0
   ```

3. Deploy:
   ```bash
   heroku login
   heroku create pdi-risk-dashboard
   git push heroku main
   ```

## Data Management

- **Development:** Use sample data in `sample_data/`
- **Production:** Upload data via the "Upload Data" dashboard page
- **Backup:** Regular backups of `/data` directory recommended

## Security Considerations

- ✅ Store secrets in `.streamlit/secrets.toml` (not in version control)
- ✅ Restrict file upload size (configured to 200MB)
- ✅ Use environment variables via `.env` file
- ✅ Consider running behind a proxy (nginx/Apache) in production
- ✅ Enable HTTPS if accessible over network
- ✅ Restrict access with authentication if needed

## Performance Tuning

For large datasets:

1. Enable caching:
   ```python
   @st.cache_data
   def load_csv(filename):
       # loading code
   ```

2. Increase Streamlit cache size in `.streamlit/config.toml`:
   ```toml
   [client]
   maxCacheMessageSize = 200
   ```

3. Use data aggregation/filtering before visualization

## Monitoring

- **Logs:** Check Streamlit terminal output for errors
- **Uptime:** Use monitoring service (StatusPage.io, UptimeRobot)
- **Performance:** Monitor response times during peak usage

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce data range, add pagination |
| Slow uploads | Increase `maxUploadSize` in config.toml |
| CORS errors | Ensure data origins are trusted |
| Missing data files | Check `/data` directory permissions |

## Support

For issues or questions:
1. Check the [Streamlit documentation](https://docs.streamlit.io)
2. Review the README.md for usage guides
3. Check the "Data Health" page to validate data integrity
