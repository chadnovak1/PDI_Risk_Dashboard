#!/bin/bash
# PDI Risk Dashboard - Setup Script

set -e  # Exit on any error

echo "================================"
echo "PDI Risk Dashboard - Setup"
echo "================================"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

echo ""
echo "🔗 Activating virtual environment..."
source venv/bin/activate

echo "📥 Installing dependencies..."
pip install --quiet -r requirements.txt
echo "✓ Dependencies installed"

echo ""
echo "================================"
echo "✅ Setup Complete!"
echo "================================"
echo ""
echo "To start the dashboard, run:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
echo ""
echo "Or use the start script:"
echo "  ./start.sh"
echo ""
