# Migration Guide: From Legacy to Clean Architecture

## Quick Migration (Recommended)

### Step 1: Backup Current System
```bash
# Create backup of current main.py
cp main.py main_legacy_backup.py
```

### Step 2: Replace main.py with Clean Architecture
```bash
# Replace main.py with the new clean architecture
cp main_new.py main.py
```

### Step 3: Test the Migration
```bash
# Start the server to test
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Step 4: Verify All Endpoints Work
Test these endpoints to ensure backward compatibility:
- `POST /process-video-s3-async/` (legacy video processing)
- `POST /apply-video-effect/` (new single effects)
- `POST /apply-combined-effects/` (new combined effects)
- `GET /job-status/{job_id}` (job status)
- `GET /performance/status` (system status)
- `GET /health` (health check)

---

## Gradual Migration (Conservative Approach)

If you prefer to migrate gradually:

### Step 1: Deploy Both Versions Side-by-Side

**Option A: Different Ports**
```bash
# Terminal 1: Run legacy version
python3 -m uvicorn main:app --reload --port 8000

# Terminal 2: Run new version  
python3 -m uvicorn main_new:app --reload --port 8001
```

**Option B: Different Endpoints**
```python
# In main.py, add new endpoints alongside existing ones
from main_new import (
    VideoProcessingController,
    SystemController
)

# Add new endpoints with /v2/ prefix
@app.post("/v2/apply-video-effect/")
async def apply_video_effect_v2(request: VideoEffectRequest):
    controller = VideoProcessingController()
    return await controller.apply_video_effects(request.dict())
```

### Step 2: Test New Architecture
- Test all new endpoints thoroughly
- Compare performance and functionality
- Ensure all dependencies work correctly

### Step 3: Switch Traffic
- Update clients to use new endpoints
- Monitor for any issues
- Keep legacy endpoints as fallback

### Step 4: Complete Migration
Once confident, replace main.py:
```bash
mv main.py main_legacy.py
mv main_new.py main.py
```

---

## Troubleshooting Common Issues

### Issue 1: Import Errors
**Problem**: Missing dependencies for new architecture
**Solution**: 
```bash
# Install any missing dependencies
pip install pydantic fastapi uvicorn asyncio
```

### Issue 2: Configuration Errors  
**Problem**: Environment variables not loaded
**Solution**:
```bash
# Ensure .env file exists and contains:
DATABASE_URL=your_database_url
AWS_ACCESS_KEY=your_aws_key
AWS_SECRET_KEY=your_aws_secret
AWS_REGION=your_region
S3_BUCKET=your_bucket
MAX_CONCURRENT_PROCESSES=4
```

### Issue 3: Service Initialization Failures
**Problem**: Dependency injection container fails to initialize
**Solution**: Check logs for specific service failures:
```bash
tail -f logs/video_processor.log
```

### Issue 4: Database Connection Issues
**Problem**: Database manager can't connect
**Solution**: Verify database credentials and connection string in .env

---

## Verification Checklist

After migration, verify these work:

### ✅ API Endpoints
- [ ] `POST /process-video-s3-async/` - Legacy processing works
- [ ] `POST /apply-video-effect/` - New effects work  
- [ ] `POST /apply-combined-effects/` - Combined effects work
- [ ] `GET /job-status/{job_id}` - Status tracking works
- [ ] `GET /performance/status` - Performance monitoring works
- [ ] `GET /health` - Health check responds

### ✅ System Components
- [ ] Database connection established
- [ ] S3 integration working
- [ ] Video processing functional
- [ ] Hardware acceleration detected
- [ ] Resource monitoring active
- [ ] Logging system operational

### ✅ Background Processing
- [ ] Jobs process in background
- [ ] Status updates correctly
- [ ] Files uploaded to S3
- [ ] Temporary files cleaned up
- [ ] Resource limits respected

---

## Rolling Back (If Needed)

If issues occur, you can quickly rollback:

```bash
# Stop the new version
# Restore the backup
cp main_legacy_backup.py main.py

# Restart with legacy version
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## Performance Comparison

After migration, you should see:

### Improvements:
- **Faster startup time** (lazy service initialization)
- **Lower memory usage** (better resource management) 
- **Better error handling** (structured logging and recovery)
- **Improved monitoring** (comprehensive health checks)

### Metrics to Monitor:
```bash
# Check system resources
htop

# Check application logs
tail -f logs/video_processor.log

# Test endpoint performance
curl -X GET http://localhost:8000/performance/status
```

---

## Next Steps After Migration

1. **Update Documentation**: Update API docs with new endpoints
2. **Client Updates**: Update any clients to use new endpoints
3. **Monitoring Setup**: Set up monitoring for the new architecture
4. **Remove Legacy Code**: After confidence period, remove old files

---

## Support

If you encounter issues during migration:

1. **Check Logs**: Look in `logs/video_processor.log` for errors
2. **Verify Config**: Ensure all environment variables are set
3. **Test Dependencies**: Run `python3 -m py_compile main.py` to check syntax
4. **Gradual Approach**: Use side-by-side deployment for testing

The new architecture is backward compatible, so all existing clients should continue working without changes.