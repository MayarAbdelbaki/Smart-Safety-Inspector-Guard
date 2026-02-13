#!/bin/bash
# Setup script for AIot Autocar Prime
# Run this on the autocar to configure the webhook URL

# Your PC's IP address
PC_IP="192.168.56.1"
PORT="5000"

# Set webhook URL
export WEBHOOK_URL="http://${PC_IP}:${PORT}/webhook"

echo "Webhook URL configured: $WEBHOOK_URL"
echo ""
echo "To make this permanent, add this line to your ~/.bashrc:"
echo "export WEBHOOK_URL=\"http://${PC_IP}:${PORT}/webhook\""
echo ""
echo "Now you can run: python3 face_capture.py"

