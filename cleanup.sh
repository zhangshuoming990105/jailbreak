#!/bin/bash

# Cleanup script for jailbreak experiment artifacts
# Removes CSV files and log files from the current directory

echo "Cleaning up jailbreak experiment artifacts..."

# Remove CSV files (question-answer results)
echo "Removing CSV files..."
rm -f *.csv

# Remove log files
echo "Removing log files..."
rm -f *.log

# Count removed files
csv_removed=$(ls *.csv 2>/dev/null | wc -l)
log_removed=$(ls *.log 2>/dev/null | wc -l)

if [ $csv_removed -eq 0 ] && [ $log_removed -eq 0 ]; then
    echo "No CSV or log files found to remove."
else
    echo "Cleanup complete!"
fi