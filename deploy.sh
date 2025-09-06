#!/bin/bash
set -e

# EC2 Deployment Script for FastAPI Video Processing Service
echo "Starting deployment on EC2..."

# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python 3.11 and pip
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Install FFmpeg with hardware acceleration support
sudo apt install -y ffmpeg

# Install MySQL client
sudo apt install -y mysql-client-core-8.0

# Install git if not present
sudo apt install -y git

# Clone repository
PROJECT_DIR="/opt/video-processor"
sudo mkdir -p $PROJECT_DIR
sudo chown -R $USER:$USER $PROJECT_DIR
cd $PROJECT_DIR

if [ -d ".git" ]; then
    echo "Repository exists, pulling latest changes..."
    git pull origin master
else
    echo "Cloning repository..."
    git clone https://github.com/your-username/your-repo.git .
fi

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

echo "==================================="
echo "Deployment setup complete!"
echo "==================================="
echo "Next steps:"
echo "1. Edit /opt/video-processor/.env with your AWS and database credentials"
echo "2. Run: sudo ./setup-service.sh to create systemd service"
echo "3. Start service: sudo systemctl start video-processor"
echo "==================================="