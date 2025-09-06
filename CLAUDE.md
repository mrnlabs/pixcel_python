# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a FastAPI-based video processing service that provides hardware-accelerated video editing through FFmpeg. The service processes videos with various effects, manages AWS S3 uploads/downloads, and tracks jobs in a MySQL database.

## Architecture

The application consists of five main Python modules:

- **main.py**: FastAPI application server with endpoints for video processing jobs, database operations using SQLAlchemy with MySQL
- **video_editor.py**: Core FFmpeg wrapper with hardware acceleration (NVENC, QuickSync, VideoToolbox), memory optimization, and video effects processing
- **s3_handler.py**: AWS S3 client for downloading source videos and uploading processed results with streaming and parallel transfers
- **resource_manager.py**: System resource monitoring and semaphore-based concurrency control to prevent memory overload
- **hardware_accelerated_ffmpeg.py**: Hardware acceleration detection utilities and processing time estimation

The system uses a job queue pattern where:
1. Jobs are submitted via POST `/process-video/` with S3 URLs
2. Background tasks process videos with resource management
3. Status is tracked in MySQL and queryable via GET `/job-status/{job_id}`

## Development Commands

### Environment Setup
```bash
# Activate Python virtual environment
source bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Server
```bash
# Start FastAPI with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or use the local uvicorn binary
./bin/uvicorn main:app --reload
```

### Database
The application auto-creates tables on startup. Connection is configured via DATABASE_URL in `.env` or defaults to the hardcoded MySQL connection string.

## Key API Endpoints

- **POST /process-video/**: Submit video processing job with parameters like video_url, audio_url, overlay_url, effects, timing settings
- **GET /job-status/{job_id}**: Check processing status of a job
- **POST /hardware-info/**: Get available hardware acceleration capabilities
- **GET /health**: Health check with current resource usage metrics

## Environment Variables

Required in `.env`:
- **AWS_ACCESS_KEY**, **AWS_SECRET_KEY**, **AWS_REGION**: AWS credentials for S3 operations
- **S3_BUCKET**: Target bucket for processed videos (default: pixcelcapetown)
- **DATABASE_URL**: MySQL connection string 
- **MAX_CONCURRENT_PROCESSES**: Parallel video processing limit (default: 4)

## Video Processing Capabilities

The system supports:
- Hardware acceleration auto-detection and usage when available
- Effects: slow motion, fade in/out, blur, stabilization
- Audio replacement and video overlay
- Adaptive quality based on available system resources
- Memory-optimized processing for large video files