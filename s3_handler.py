import boto3
import os
import concurrent.futures
import threading
import tempfile
from datetime import datetime

class OptimizedS3Handler:
    def __init__(self):
        """Initialize S3 client with optimized settings"""
        # Get AWS credentials from environment
        AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
        AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
        AWS_REGION = os.getenv('AWS_REGION', 'af-south-1')
        self.S3_BUCKET = os.getenv('S3_BUCKET', 'picxel-bucket')
        
        try:
            # Create client with minimal memory footprint and improved performance
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY,
                region_name=AWS_REGION,
                # Memory optimization options
                config=boto3.session.Config(
                    max_pool_connections=20,  # Increased for parallel downloads
                    retries={'max_attempts': 3},  # Reasonable retry count
                    use_dualstack_endpoint=False,  # Simpler networking
                    # S3 specific optimizations
                    s3={'addressing_style': 'path'}, 
                    # Use TCP keepalive
                    connect_timeout=5,
                    read_timeout=60  # Longer timeout for big files
                )
            )
            # Verify connection
            self.s3_client.list_buckets()
            print("Successfully connected to S3 with optimized client")
        except Exception as e:
            print(f"Error initializing S3 client: {str(e)}")
            raise

    def download_file(self, url: str, file_type: str) -> str:
        """
        Download file from S3 with memory optimization.
        Uses streaming download to minimize memory usage with improved throughput.
        
        Args:
            url (str): S3 URL or key of the file to download
            file_type (str): File extension (e.g., 'mp4', 'aac')
            
        Returns:
            str: Path to the downloaded temporary file
        """
        try:
            # Extract the key from the S3 URL
            if url.startswith('http'):
                # Remove the bucket and AWS domain to get the key
                key = url.split('.amazonaws.com/')[-1]
            else:
                key = url

            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}')
            temp_path = temp_file.name
            temp_file.close()

            print(f"Downloading {key} from S3 with optimized streaming...")
            
            # OPTIMIZATION: Use multipart download for large files
            try:
                # Get object info to check size
                response = self.s3_client.head_object(Bucket=self.S3_BUCKET, Key=key)
                file_size = response.get('ContentLength', 0)
                
                if file_size > 50 * 1024 * 1024:  # 50MB
                    # For large files, use multipart download
                    print(f"Large file detected ({file_size/(1024*1024):.1f} MB). Using multipart download.")
                    
                    # Create a thread-safe counter for progress tracking
                    bytes_transferred = 0
                    bytes_lock = threading.Lock()
                    
                    def download_part(part_info):
                        nonlocal bytes_transferred
                        part_num, start_byte, end_byte = part_info
                        
                        # Download this part
                        part_response = self.s3_client.get_object(
                            Bucket=self.S3_BUCKET,
                            Key=key,
                            Range=f"bytes={start_byte}-{end_byte}"
                        )
                        
                        # Read part data
                        part_data = part_response['Body'].read()
                        
                        # Write to the correct position in the file
                        with open(temp_path, 'r+b') as f:
                            f.seek(start_byte)
                            f.write(part_data)
                        
                        # Update progress
                        with bytes_lock:
                            bytes_transferred += len(part_data)
                            percent = (bytes_transferred / file_size) * 100
                            if part_num % 4 == 0:  # Only log every few parts
                                print(f"Download progress: {percent:.1f}% ({bytes_transferred/(1024*1024):.1f} MB)")
                    
                    # Determine optimal part size
                    part_size = 8 * 1024 * 1024  # 8MB parts
                    num_parts = (file_size + part_size - 1) // part_size  # Round up
                    
                    # Create empty file of the required size
                    with open(temp_path, 'wb') as f:
                        f.seek(file_size - 1)
                        f.write(b'\0')
                    
                    # Create download tasks
                    download_tasks = []
                    for i in range(int(num_parts)):
                        start_byte = i * part_size
                        end_byte = min(start_byte + part_size - 1, file_size - 1)
                        download_tasks.append((i, start_byte, end_byte))
                    
                    # Use thread pool to download parts concurrently
                    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                        executor.map(download_part, download_tasks)
                
                else:
                    # For smaller files, use simpler streaming approach
                    s3_object = self.s3_client.get_object(Bucket=self.S3_BUCKET, Key=key)
                    
                    # Streaming write with small chunks
                    with open(temp_path, 'wb') as f:
                        # Stream in 1MB chunks
                        chunk_size = 1024 * 1024  # 1MB
                        body = s3_object['Body']
                        
                        bytes_read = 0
                        chunk = body.read(chunk_size)
                        while chunk:
                            f.write(chunk)
                            bytes_read += len(chunk)
                            chunk = body.read(chunk_size)
                            
                            # Log progress occasionally
                            if file_size > 0 and bytes_read % (5 * chunk_size) == 0:
                                percent = (bytes_read / file_size) * 100
                                print(f"Download progress: {percent:.1f}% ({bytes_read/(1024*1024):.1f} MB)")
            
            except Exception as e:
                print(f"Error during optimized download: {str(e)}, falling back to simple method")
                
                # Fallback to simple download
                s3_object = self.s3_client.get_object(Bucket=self.S3_BUCKET, Key=key)
                
                with open(temp_path, 'wb') as f:
                    f.write(s3_object['Body'].read())
            
            print(f"Successfully downloaded to {temp_path}")
            return temp_path

        except Exception as e:
            error_msg = f"Failed to download file from S3: {str(e)}"
            print(error_msg)
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=error_msg)

    def upload_file(self, file_path: str) -> str:
        """
        Upload file to S3 with memory optimization and improved speeds.
        
        Args:
            file_path (str): Path to the file to upload
            
        Returns:
            str: S3 URL of the uploaded file
        """
        try:
            # Generate a unique key for the file
            file_name = os.path.basename(file_path)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            key = f"processed_videos/{timestamp}_{file_name}"
            
            # Get file size to determine upload strategy
            file_size = os.path.getsize(file_path)
            
            print(f"Uploading {file_path} ({file_size} bytes) to S3 with optimized settings...")
            
            if file_size > 50 * 1024 * 1024:  # For files larger than 50MB
                # OPTIMIZATION: Use multipart upload with improved settings
                from boto3.s3.transfer import TransferConfig
                
                # Configure with optimal performance settings
                transfer_config = TransferConfig(
                    multipart_threshold=8 * 1024 * 1024,  # 8MB
                    max_concurrency=5,  # Increased for better parallelism
                    multipart_chunksize=16 * 1024 * 1024,  # 16MB chunks for faster uploads
                    use_threads=True
                )
                
                # Show progress during upload
                class ProgressTracker:
                    def __init__(self, file_size):
                        self.file_size = file_size
                        self.uploaded_bytes = 0
                        self.lock = threading.Lock()
                        self.last_logged = 0
                    
                    def __call__(self, bytes_transferred):
                        with self.lock:
                            self.uploaded_bytes += bytes_transferred
                            progress = (self.uploaded_bytes / self.file_size) * 100
                            
                            # Only log progress every 5%
                            if progress >= self.last_logged + 5:
                                print(f"Upload progress: {progress:.1f}% ({self.uploaded_bytes/(1024*1024):.1f} MB)")
                                self.last_logged = progress - (progress % 5)
                
                progress_tracker = ProgressTracker(file_size)
                
                with open(file_path, 'rb') as file_data:
                    self.s3_client.upload_fileobj(
                        file_data,
                        Bucket=self.S3_BUCKET,
                        Key=key,
                        Config=transfer_config,
                        Callback=progress_tracker
                    )
            else:
                # For smaller files, standard upload is fine
                self.s3_client.upload_file(
                    Filename=file_path,
                    Bucket=self.S3_BUCKET,
                    Key=key
                )
            
            # Return the URL
            AWS_REGION = self.s3_client.meta.region_name
            url = f"https://{self.S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"
            print(f"Successfully uploaded to {url}")
            return url

        except Exception as e:
            error_msg = f"Failed to upload file to S3: {str(e)}"
            print(error_msg)
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=error_msg)