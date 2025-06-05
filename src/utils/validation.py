"""
Validation utilities for document ingestion.
"""

import os
try:
    import tiktoken
except ModuleNotFoundError:  # pragma: no cover
    tiktoken = None
from typing import Tuple, Optional
from pathlib import Path


# Constants
MAX_FILE_SIZE_MB = 150
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_TOKEN_COUNT = 1_000_000

# Supported file extensions
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.md', '.txt', '.markdown'}


def validate_file_size(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that file size is within acceptable limits.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        file_size = os.path.getsize(file_path)
        
        if file_size > MAX_FILE_SIZE_BYTES:
            size_mb = file_size / (1024 * 1024)
            return False, f"File size ({size_mb:.1f} MB) exceeds maximum allowed size ({MAX_FILE_SIZE_MB} MB)"
        
        if file_size == 0:
            return False, "File is empty"
            
        return True, None
        
    except OSError as e:
        return False, f"Error reading file: {str(e)}"


def validate_file_extension(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that file has a supported extension.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext not in SUPPORTED_EXTENSIONS:
        supported = ', '.join(sorted(SUPPORTED_EXTENSIONS))
        return False, f"Unsupported file extension '{file_ext}'. Supported: {supported}"
    
    return True, None


def validate_file(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Comprehensive file validation.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if file exists
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"
    
    if not os.path.isfile(file_path):
        return False, f"Path is not a file: {file_path}"
    
    # Validate extension
    is_valid, error = validate_file_extension(file_path)
    if not is_valid:
        return is_valid, error
    
    # Validate size
    return validate_file_size(file_path)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Text to count tokens for
        encoding_name: Tiktoken encoding to use (default: cl100k_base for GPT-4)
        
    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to simple word-based estimation if tiktoken fails
        return int(len(text.split()) * 1.3)  # Rough approximation


def validate_token_count(text: str) -> Tuple[bool, Optional[str], int]:
    """
    Validate that text doesn't exceed maximum token count.
    
    Args:
        text: Text to validate
        
    Returns:
        Tuple of (is_valid, error_message, token_count)
    """
    token_count = count_tokens(text)
    
    if token_count > MAX_TOKEN_COUNT:
        return False, f"Token count ({token_count:,}) exceeds maximum allowed ({MAX_TOKEN_COUNT:,})", token_count
    
    return True, None, token_count


def estimate_token_count_from_file_size(file_size_bytes: int) -> int:
    """
    Estimate token count from file size (rough approximation).
    
    Args:
        file_size_bytes: File size in bytes
        
    Returns:
        Estimated token count
    """
    # Rough estimation: 1 token ≈ 4 characters, 1 character ≈ 1 byte
    return file_size_bytes // 4


def validate_file_before_processing(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Quick validation before processing to catch obvious issues.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Basic file validation
    is_valid, error = validate_file(file_path)
    if not is_valid:
        return is_valid, error
    
    # Estimate if file might exceed token limit
    file_size = os.path.getsize(file_path)
    estimated_tokens = estimate_token_count_from_file_size(file_size)
    
    if estimated_tokens > MAX_TOKEN_COUNT:
        return False, f"File likely exceeds token limit (estimated {estimated_tokens:,} tokens, max {MAX_TOKEN_COUNT:,})"
    
    return True, None