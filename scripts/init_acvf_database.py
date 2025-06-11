#!/usr/bin/env python3
"""
ACVF Database Initialization Script.

This script creates the database tables required for ACVF (Adversarial Cross-Validation Framework)
persistence including debate rounds, judge scores, and model assignments.
"""

import sys
import os
from pathlib import Path

# Add src directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.database import DatabaseManager, create_tables, drop_tables
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Initialize the ACVF database."""
    try:
        logger.info("Starting ACVF database initialization...")
        
        # Check if DATABASE_URL is set
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            logger.info(f"Using database URL: {database_url}")
        else:
            logger.info("No DATABASE_URL set, using default SQLite database")
        
        # Initialize database manager
        db_manager = DatabaseManager()
        
        # Create tables
        logger.info("Creating ACVF database tables...")
        db_manager.initialize_database()
        
        logger.info("✅ ACVF database initialization completed successfully!")
        logger.info("Created tables:")
        logger.info("  - model_assignments: Store LLM model configurations for debates")
        logger.info("  - debate_rounds: Store individual ACVF debate rounds")
        logger.info("  - debate_arguments: Store challenger/defender arguments")
        logger.info("  - judge_scores: Store judge verdicts and detailed scoring")
        logger.info("  - acvf_sessions: Store complete ACVF debate session results")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize ACVF database: {str(e)}")
        return False


def reset_database():
    """Reset the ACVF database (drop and recreate all tables)."""
    try:
        logger.warning("⚠️  RESETTING ACVF DATABASE - This will delete all existing data!")
        
        response = input("Are you sure you want to reset the database? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Database reset cancelled.")
            return False
        
        # Reset database
        db_manager = DatabaseManager()
        db_manager.reset_database()
        
        logger.info("✅ ACVF database reset completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to reset ACVF database: {str(e)}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize ACVF database")
    parser.add_argument("--reset", action="store_true", 
                       help="Reset database (drop and recreate all tables)")
    
    args = parser.parse_args()
    
    if args.reset:
        success = reset_database()
    else:
        success = main()
    
    sys.exit(0 if success else 1) 