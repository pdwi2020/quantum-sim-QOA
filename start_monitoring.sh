#!/bin/bash

echo "🚀 Starting Quantum TSP Results Processing"
echo "This script will:"
echo "  - Monitor job status every 30 seconds"
echo "  - Fetch logs when jobs complete"
echo "  - Generate CSVs from experiment data"
echo "  - Create comparison graphs"
echo "  - Generate a comprehensive summary report"
echo ""
echo "Results will be saved in: ./final_paper_results/"
echo ""
echo "Press Ctrl+C to stop monitoring (safe to restart)"
echo ""

python3 process_final_results.py
