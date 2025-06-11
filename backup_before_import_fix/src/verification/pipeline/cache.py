"""
Caching system for verification results to avoid redundant analysis.
"""

import asyncio
import hashlib
import logging
import time
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from ...models.verification import VerificationChainResult

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry for verification results."""
    result: VerificationChainResult
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    content_hash: str = ""
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if the cache entry is expired."""
        expiry_time = self.created_at + timedelta(seconds=ttl_seconds)
        return datetime.utcnow() > expiry_time
    
    def touch(self) -> None:
        """Update last accessed time and increment access count."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


class VerificationCache:
    """
    Caching system for verification results with TTL and LRU eviction.
    """
    
    def __init__(self, ttl_seconds: int = 3600, size_limit: int = 1000):
        """
        Initialize the verification cache.
        
        Args:
            ttl_seconds: Time-to-live for cache entries in seconds
            size_limit: Maximum number of entries to store
        """
        self.ttl_seconds = ttl_seconds
        self.size_limit = size_limit
        self.cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired": 0
        }
        
        logger.info(f"Initialized VerificationCache with TTL={ttl_seconds}s, size_limit={size_limit}")
    
    def _generate_cache_key(self, document_id: str, content: str) -> str:
        """
        Generate a cache key for the document and content.
        
        Args:
            document_id: Document identifier
            content: Document content
            
        Returns:
            Cache key string
        """
        # Create a hash of the content to detect changes
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        # Combine document ID and content hash
        cache_key = f"{document_id}:{content_hash}"
        return cache_key
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate a hash of the content for change detection."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    async def get(self, document_id: str, content: str) -> Optional[VerificationChainResult]:
        """
        Get a cached verification result.
        
        Args:
            document_id: Document identifier
            content: Document content
            
        Returns:
            Cached VerificationChainResult or None if not found/expired
        """
        cache_key = self._generate_cache_key(document_id, content)
        
        async with self._lock:
            entry = self.cache.get(cache_key)
            
            if entry is None:
                self.stats["misses"] += 1
                logger.debug(f"Cache miss for document {document_id}")
                return None
            
            # Check if entry is expired
            if entry.is_expired(self.ttl_seconds):
                self.stats["expired"] += 1
                del self.cache[cache_key]
                logger.debug(f"Cache entry expired for document {document_id}")
                return None
            
            # Update access statistics
            entry.touch()
            self.stats["hits"] += 1
            
            logger.debug(f"Cache hit for document {document_id} (accessed {entry.access_count} times)")
            return entry.result
    
    async def set(self, document_id: str, content: str, result: VerificationChainResult) -> None:
        """
        Store a verification result in the cache.
        
        Args:
            document_id: Document identifier
            content: Document content
            result: Verification result to cache
        """
        cache_key = self._generate_cache_key(document_id, content)
        content_hash = self._generate_content_hash(content)
        
        async with self._lock:
            # Check if we need to evict entries
            if len(self.cache) >= self.size_limit:
                await self._evict_entries()
            
            # Create cache entry
            entry = CacheEntry(
                result=result,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                content_hash=content_hash
            )
            
            self.cache[cache_key] = entry
            logger.debug(f"Cached verification result for document {document_id}")
    
    async def _evict_entries(self) -> None:
        """Evict old entries using LRU strategy."""
        if not self.cache:
            return
        
        # Calculate how many entries to evict (25% of size limit)
        evict_count = max(1, self.size_limit // 4)
        
        # Sort by last accessed time (oldest first)
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Evict oldest entries
        for i in range(min(evict_count, len(sorted_entries))):
            cache_key, _ = sorted_entries[i]
            del self.cache[cache_key]
            self.stats["evictions"] += 1
        
        logger.debug(f"Evicted {min(evict_count, len(sorted_entries))} cache entries")
    
    async def invalidate(self, document_id: str) -> int:
        """
        Invalidate all cache entries for a specific document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Number of entries invalidated
        """
        async with self._lock:
            keys_to_remove = [
                key for key in self.cache.keys()
                if key.startswith(f"{document_id}:")
            ]
            
            for key in keys_to_remove:
                del self.cache[key]
            
            logger.debug(f"Invalidated {len(keys_to_remove)} cache entries for document {document_id}")
            return len(keys_to_remove)
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            entry_count = len(self.cache)
            self.cache.clear()
            self.stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "expired": 0
            }
            logger.info(f"Cleared {entry_count} cache entries")
    
    async def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries.
        
        Returns:
            Number of expired entries removed
        """
        async with self._lock:
            expired_keys = []
            
            for key, entry in self.cache.items():
                if entry.is_expired(self.ttl_seconds):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
                self.stats["expired"] += 1
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
            
            return len(expired_keys)
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        async with self._lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
            
            return {
                "size": len(self.cache),
                "size_limit": self.size_limit,
                "ttl_seconds": self.ttl_seconds,
                "hit_rate": hit_rate,
                "statistics": self.stats.copy(),
                "memory_usage": {
                    "entry_count": len(self.cache),
                    "average_access_count": sum(entry.access_count for entry in self.cache.values()) / len(self.cache) if self.cache else 0
                }
            }
    
    async def get_health(self) -> Dict[str, Any]:
        """
        Get health status of the cache.
        
        Returns:
            Health status information
        """
        stats = await self.get_stats()
        
        # Determine health status
        size_usage = stats["size"] / stats["size_limit"] if stats["size_limit"] > 0 else 0
        
        if size_usage > 0.9:
            status = "warning"
            message = "Cache is nearly full"
        elif stats["hit_rate"] < 0.2 and stats["statistics"]["hits"] + stats["statistics"]["misses"] > 10:
            status = "warning"
            message = "Low cache hit rate"
        else:
            status = "healthy"
            message = "Cache operating normally"
        
        return {
            "status": status,
            "message": message,
            "size_usage_percent": size_usage * 100,
            "hit_rate_percent": stats["hit_rate"] * 100,
            "stats": stats
        }
    
    async def shutdown(self) -> None:
        """Shutdown the cache and cleanup resources."""
        await self.clear()
        logger.info("VerificationCache shutdown complete") 