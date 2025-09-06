#!/bin/bash
set -e

echo "Setting up systemd service..."

# Copy service file
sudo cp video-processor.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable video-processor

echo "Service setup complete!"
echo "Commands:"
echo "  Start:   sudo systemctl start video-processor"
echo "  Stop:    sudo systemctl stop video-processor"
echo "  Status:  sudo systemctl status video-processor"
echo "  Logs:    sudo journalctl -u video-processor -f"