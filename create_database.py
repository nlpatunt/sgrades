# create_database.py - Simple script to create database with correct schema

import sqlite3
import os

def create_database():
    """Create database with correct schema"""
    
    # Remove old database if it exists
    if os.path.exists("besesr.db"):
        os.remove("besesr.db")
        print("🗑️ Removed old database")
    
    # Create new database
    conn = sqlite3.connect("besesr.db")
    cursor = conn.cursor()
    
    print("🔄 Creating new database tables...")
    
    # Create datasets table
    cursor.execute("""
        CREATE TABLE datasets (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            huggingface_id TEXT,
            essay_count INTEGER DEFAULT 0,
            avg_essay_length REAL DEFAULT 0.0,
            score_range_min REAL DEFAULT 0.0,
            score_range_max REAL DEFAULT 6.0,
            is_active BOOLEAN DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("   ✅ Created datasets table")
    
    # Create output_submissions table with ALL required columns
    cursor.execute("""
        CREATE TABLE output_submissions (
            id INTEGER PRIMARY KEY,
            dataset_name TEXT NOT NULL,
            submitter_name TEXT NOT NULL,
            submitter_email TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_format TEXT DEFAULT 'csv',
            status TEXT DEFAULT 'submitted',
            description TEXT,
            evaluation_result TEXT,
            error_message TEXT,
            submission_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            processing_time REAL
        )
    """)
    print("   ✅ Created output_submissions table")
    
    # Create evaluation_results table
    cursor.execute("""
        CREATE TABLE evaluation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            submission_id INTEGER NOT NULL,
            dataset_name TEXT NOT NULL,
            quadratic_weighted_kappa REAL,
            pearson_correlation REAL,
            spearman_correlation REAL,
            mean_absolute_error REAL,
            root_mean_squared_error REAL,
            f1_score REAL,
            accuracy REAL,
            essays_evaluated INTEGER DEFAULT 0,
            evaluation_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            evaluation_duration REAL,
            status TEXT DEFAULT 'completed',
            error_message TEXT,
            detailed_metrics TEXT,
            FOREIGN KEY (submission_id) REFERENCES output_submissions (id)
        )
    """)
    print("   ✅ Created evaluation_results table")
    
    # Create essays table
    cursor.execute("""
        CREATE TABLE essays (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            essay_id TEXT UNIQUE NOT NULL,
            dataset_name TEXT NOT NULL,
            essay_text TEXT NOT NULL,
            prompt TEXT,
            holistic_score REAL,
            content_score REAL,
            organization_score REAL,
            style_score REAL,
            grammar_score REAL,
            word_count INTEGER,
            sentence_count INTEGER,
            paragraph_count INTEGER,
            grade_level TEXT,
            essay_attributes TEXT,
            created_time DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("   ✅ Created essays table")
    
    # Insert some sample datasets
    sample_datasets = [
        ('ASAP-AES', 'ASAP Automated Essay Scoring', 'asap-aes', 0, 0.0, 0.0, 6.0),
        ('ASAP-SAS', 'ASAP Short Answer Scoring', 'asap-sas', 0, 0.0, 0.0, 3.0),
        ('rice_chem', 'Rice Chemistry Dataset', 'rice-chem', 0, 0.0, 0.0, 5.0),
    ]
    
    cursor.executemany("""
        INSERT INTO datasets (name, description, huggingface_id, essay_count, avg_essay_length, score_range_min, score_range_max)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, sample_datasets)
    print("   ✅ Added sample datasets")
    
    conn.commit()
    conn.close()
    
    print("✅ Database created successfully!")
    
    # Verify the schema
    verify_schema()

def verify_schema():
    """Verify that all tables have the correct columns"""
    conn = sqlite3.connect("besesr.db")
    cursor = conn.cursor()
    
    print("\n📋 Verifying database schema...")
    
    # Check output_submissions table
    cursor.execute("PRAGMA table_info(output_submissions)")
    columns = cursor.fetchall()
    column_names = [col[1] for col in columns]
    
    print(f"   output_submissions columns: {column_names}")
    
    required_columns = ['id', 'dataset_name', 'submitter_name', 'submitter_email', 
                       'file_path', 'file_format', 'status', 'description', 
                       'evaluation_result', 'error_message', 'submission_time', 'processing_time']
    
    missing = [col for col in required_columns if col not in column_names]
    if missing:
        print(f"   ❌ Missing columns: {missing}")
    else:
        print(f"   ✅ All required columns present")
    
    conn.close()

if __name__ == "__main__":
    create_database()