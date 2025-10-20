#!/bin/bash
# Setup script for Deribit options scraper cron job
# This will run the scraper daily at 12:00 UTC

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
VENV_PATH="$PROJECT_DIR/.venv"
SCRAPER_SCRIPT="$SCRIPT_DIR/deribit_options_scraper.py"

# Create wrapper script for cron
WRAPPER_SCRIPT="$SCRIPT_DIR/run_scraper.sh"

cat > "$WRAPPER_SCRIPT" << 'EOF'
#!/bin/bash
# Wrapper script to run the Deribit scraper with virtual environment

cd "$(dirname "$0")/../.."
source .venv/bin/activate
python scripts/data_collection/deribit_options_scraper.py
EOF

chmod +x "$WRAPPER_SCRIPT"

echo "Scraper setup complete!"
echo ""
echo "To run the scraper manually:"
echo "  $WRAPPER_SCRIPT"
echo ""
echo "To set up daily automatic collection at 12:00 UTC, add this to your crontab:"
echo "  crontab -e"
echo ""
echo "Then add this line:"
echo "  0 12 * * * $WRAPPER_SCRIPT >> $PROJECT_DIR/data/raw/scraper_cron.log 2>&1"
echo ""
echo "This will run daily at 12:00 UTC and log output to scraper_cron.log"
echo ""
echo "To verify cron job:"
echo "  crontab -l"
