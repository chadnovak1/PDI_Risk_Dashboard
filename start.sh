#!/bin/bash
# PDI Risk Dashboard - Start Script

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Error: Virtual environment not found"
    echo "Please run setup.sh first:"
    echo "  ./setup.sh"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Error: Streamlit not found in virtual environment"
    echo "Please run setup.sh to install dependencies:"
    echo "  ./setup.sh"
    exit 1
fi

echo "🚀 Starting PDI Risk Dashboard..."
echo ""
echo "   Local URL: http://localhost:8503"
echo "   To stop: Press Ctrl+C"
echo ""

streamlit run app.py
