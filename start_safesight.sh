#!/bin/bash

# Navigate to the application directory
cd /home/werkstudent/Desktop/DemoCase/safesight/DemoCase

# Activate the virtual environment
source /home/werkstudent/Desktop/DemoCase/safesight/.venv/bin/activate

# Run the application
echo "Starting SafeSight..."
python3 app.py

# Keep the terminal open if the app closes (for debugging)
echo "Application closed. Press Enter to exit."
read
