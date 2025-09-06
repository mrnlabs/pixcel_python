"""
Database Connection Pool Manager
Provides efficient database connection pooling, health checks, and automatic recovery.
"""
import asyncio
import databases
import sqlalchemy
from sqlalchemy.pool import QueuePool
from typing import Optional, Dict, Any
import time
from contextlib import asynccontextmanager
from logger import get_logger, log_error
from config import get_settings

class DatabaseManager:
    """
    Manages database connections with pooling, health checks, and automatic recovery.
    """
    
    def __init__(self):
        self.logger = get_logger("database_manager")
        self.settings = get_settings()
        self._database = None
        self._engine = None
        self._is_connected = False
        self._connection_attempts = 0
        self._last_health_check = 0
        self._health_check_interval = 30  # 30 seconds
        self._max_retries = 5
        self._retry_delay = 2
        
        self._init_database()
    
    def _init_database(self):
        """Initialize database connections with optimized pooling"""
        try:
            # Create engine with optimized connection pool
            self._engine = sqlalchemy.create_engine(
                self.settings.database_url,
                poolclass=QueuePool,
                pool_size=10,              # Number of connections to maintain
                max_overflow=20,           # Additional connections allowed
                pool_pre_ping=True,        # Validate connections before use
                pool_recycle=3600,         # Recycle connections after 1 hour
                pool_timeout=30,           # Timeout when getting connection from pool
                echo=self.settings.debug,  # Log SQL queries in debug mode
                connect_args={
                    "charset": "utf8mb4",
                    "autocommit": False,
                    "connect_timeout": 10,
                    "read_timeout": 30,
                    "write_timeout": 30,
                }
            )
            
            # Create databases instance with connection pooling
            self._database = databases.Database(
                self.settings.database_url,
                min_size=5,               # Minimum connections in pool
                max_size=20,              # Maximum connections in pool
                command_timeout=30,       # Command timeout
                server_settings={
                    "jit": "off"           # Disable JIT for consistent performance
                }
            )
            
            self.logger.info("Database manager initialized with connection pooling")
            
        except Exception as e:
            log_error(e, "database_manager_initialization")
            raise
    
    async def connect(self) -> bool:
        """
        Connect to database with retry logic and health checks
        
        Returns:
            bool: True if connected successfully
        """
        if self._is_connected and await self._health_check():
            return True
        
        for attempt in range(1, self._max_retries + 1):
            try:
                await self._database.connect()
                self._is_connected = True
                self._connection_attempts = 0
                
                self.logger.info("Database connected successfully", extra={
                    'attempt': attempt,
                    'pool_size': self._engine.pool.size(),
                    'checked_out_connections': self._engine.pool.checkedout()
                })
                
                return True
                
            except Exception as e:
                self._connection_attempts += 1
                
                if attempt == self._max_retries:
                    log_error(e, "database_connection_failed", 
                             attempt=attempt, max_retries=self._max_retries)
                    self._is_connected = False
                    return False
                
                self.logger.warning(f"Database connection attempt {attempt} failed, retrying...", 
                                  extra={'error': str(e), 'retry_delay': self._retry_delay})
                
                await asyncio.sleep(self._retry_delay * attempt)  # Exponential backoff
        
        return False
    
    async def disconnect(self):
        """Disconnect from database and clean up connections"""
        try:
            if self._database and self._is_connected:
                await self._database.disconnect()
                self._is_connected = False
                self.logger.info("Database disconnected successfully")
                
        except Exception as e:
            log_error(e, "database_disconnection")
    
    async def _health_check(self) -> bool:
        """
        Perform database health check
        
        Returns:
            bool: True if database is healthy
        """
        current_time = time.time()
        
        # Skip if checked recently
        if current_time - self._last_health_check < self._health_check_interval:
            return self._is_connected
        
        try:
            # Simple health check query
            result = await self._database.fetch_one("SELECT 1 as health_check")
            self._last_health_check = current_time
            
            if result and result['health_check'] == 1:
                return True
            else:
                self.logger.warning("Database health check failed - unexpected result")
                return False
                
        except Exception as e:
            self.logger.warning("Database health check failed", extra={'error': str(e)})
            self._is_connected = False
            return False
    
    async def ensure_connection(self) -> bool:
        """
        Ensure database connection is active, reconnect if necessary
        
        Returns:
            bool: True if connection is active
        """
        if not self._is_connected or not await self._health_check():
            return await self.connect()
        
        return True
    
    async def execute(self, query, values=None):
        """
        Execute a query with automatic connection management
        
        Args:
            query: SQL query or SQLAlchemy query object
            values: Query parameters
            
        Returns:
            Query result
        """
        if not await self.ensure_connection():
            raise Exception("Database connection could not be established")
        
        try:
            if values:
                return await self._database.execute(query, values)
            else:
                return await self._database.execute(query)
                
        except Exception as e:
            # Log error with query context (be careful not to log sensitive data)
            query_str = str(query)[:100] + "..." if len(str(query)) > 100 else str(query)
            log_error(e, "database_query_execution", query=query_str)
            raise
    
    async def fetch_one(self, query, values=None):
        """
        Fetch one row with automatic connection management
        
        Args:
            query: SQL query or SQLAlchemy query object
            values: Query parameters
            
        Returns:
            Single row result or None
        """
        if not await self.ensure_connection():
            raise Exception("Database connection could not be established")
        
        try:
            if values:
                return await self._database.fetch_one(query, values)
            else:
                return await self._database.fetch_one(query)
                
        except Exception as e:
            query_str = str(query)[:100] + "..." if len(str(query)) > 100 else str(query)
            log_error(e, "database_fetch_one", query=query_str)
            raise
    
    async def fetch_all(self, query, values=None):
        """
        Fetch all rows with automatic connection management
        
        Args:
            query: SQL query or SQLAlchemy query object
            values: Query parameters
            
        Returns:
            List of rows
        """
        if not await self.ensure_connection():
            raise Exception("Database connection could not be established")
        
        try:
            if values:
                return await self._database.fetch_all(query, values)
            else:
                return await self._database.fetch_all(query)
                
        except Exception as e:
            query_str = str(query)[:100] + "..." if len(str(query)) > 100 else str(query)
            log_error(e, "database_fetch_all", query=query_str)
            raise
    
    @asynccontextmanager
    async def transaction(self):
        """
        Async context manager for database transactions
        
        Usage:
            async with db_manager.transaction():
                await db_manager.execute("INSERT ...")
                await db_manager.execute("UPDATE ...")
        """
        if not await self.ensure_connection():
            raise Exception("Database connection could not be established")
        
        transaction = await self._database.transaction()
        try:
            async with transaction:
                yield transaction
                
        except Exception as e:
            log_error(e, "database_transaction")
            raise
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get database connection pool statistics
        
        Returns:
            dict: Connection pool statistics
        """
        if not self._engine:
            return {"error": "Engine not initialized"}
        
        pool = self._engine.pool
        
        return {
            "pool_size": pool.size(),
            "checked_out_connections": pool.checkedout(),
            "overflow_connections": pool.overflow(),
            "invalid_connections": pool.invalidated(),
            "is_connected": self._is_connected,
            "connection_attempts": self._connection_attempts,
            "last_health_check": self._last_health_check
        }
    
    async def cleanup_expired_jobs(self, max_age_hours: int = 24) -> int:
        """
        Clean up expired jobs from the database
        
        Args:
            max_age_hours: Maximum age of jobs to keep
            
        Returns:
            int: Number of jobs cleaned up
        """
        try:
            cutoff_time = time.time() - (max_age_hours * 3600)
            
            # This would need to be adapted to your specific job table structure
            query = """
                DELETE FROM video_processing_jobs 
                WHERE created_at < :cutoff_time 
                AND status IN ('completed', 'failed')
            """
            
            result = await self.execute(query, {"cutoff_time": cutoff_time})
            
            self.logger.info(f"Cleaned up {result} expired jobs from database")
            return result
            
        except Exception as e:
            log_error(e, "database_cleanup")
            return 0

# Global database manager instance
_db_manager = None

def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager