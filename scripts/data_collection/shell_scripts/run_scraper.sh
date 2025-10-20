#!/bin/bash
# Wrapper script to run the Deribit scraper with virtual environment

cd "$(dirname "$0")/../.."
source .venv/bin/activate
python scripts/data_collection/deribit_options_scraper.py
