# EC2 Deployment Guide

## Prerequisites

1. **EC2 Instance**: Ubuntu 20.04+ with at least 2GB RAM
2. **Security Group**: Open port 8000 for HTTP traffic
3. **IAM Role**: EC2 instance needs S3 access permissions
4. **MySQL Database**: RDS instance or external MySQL server

## Quick Deployment Steps

### 1. Connect to EC2 Instance
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

### 2. Run Deployment Script
```bash
# Make script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh
```


### 3. Configure Environment
```bash
# Edit environment variables
sudo nano /opt/video-processor/.env

# Required settings:
# - AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION
# - DATABASE_URL (MySQL connection string)
# - S3_BUCKET name
```

### 4. Set Up Service
```bash
# Make setup script executable
chmod +x setup-service.sh

# Install systemd service
sudo ./setup-service.sh

# Start the service
sudo systemctl start video-processor

# Check status
sudo systemctl status video-processor
```

## Service Management

```bash
# Start service
sudo systemctl start video-processor

# Stop service  
sudo systemctl stop video-processor

# Restart service
sudo systemctl restart video-processor

# View logs
sudo journalctl -u video-processor -f

# Check service status
sudo systemctl status video-processor
```

## Testing Deployment

```bash
# Health check
curl http://your-ec2-ip:8000/health

# Hardware info
curl -X POST http://your-ec2-ip:8000/hardware-info/
```

## Troubleshooting

### Service won't start
```bash
# Check logs
sudo journalctl -u video-processor -n 50

# Check environment file
cat /opt/video-processor/.env

# Test manually
cd /opt/video-processor
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Database connection issues
- Verify DATABASE_URL in .env
- Check MySQL server is accessible
- Test connection: `mysql -h host -u user -p database`

### AWS S3 issues
- Verify AWS credentials in .env
- Check IAM permissions for S3 access
- Test with AWS CLI: `aws s3 ls s3://your-bucket`

## Updates

To deploy updates from master branch:
```bash
cd /opt/video-processor
git pull origin master
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart video-processor
```