#!/usr/bin/env python3
"""
Import JSON results from batch testing into BESESR database
"""
import json
import sqlite3
import sys
from datetime import datetime
import os

def import_json_to_database(json_file_path, db_path="besesr.db"):
    """Import results from JSON file into BESESR database"""
    
    # Read the JSON file
    try:
        with open(json_file_path, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} results from {json_file_path}")
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return False
    
    # Connect to database
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        print(f"Connected to database: {db_path}")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return False
    
    # Check database schema
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='output_submissions';")
    if not cursor.fetchone():
        print("ERROR: output_submissions table not found in database")
        return False
    
    # Get table schema
    cursor.execute("PRAGMA table_info(output_submissions);")
    columns = [row[1] for row in cursor.fetchall()]
    print(f"Database columns: {columns}")
    
    imported_count = 0
    skipped_count = 0
    
    for i, result in enumerate(results):
        try:
            # Extract data from JSON result
            dataset_name = result.get('dataset', '')
            model_name = result.get('model', '')
            submitter_name = result.get('submitter_name', model_name)  # Use model name if submitter_name not found
            metrics = result.get('metrics', {})
            
            # Skip if essential data missing
            if not dataset_name or not submitter_name or not metrics:
                print(f"Skipping result {i+1}: Missing essential data")
                skipped_count += 1
                continue
            
            # Check if this submission already exists
            cursor.execute("""
                SELECT id FROM output_submissions 
                WHERE submitter_name = ? AND dataset_name = ?
            """, (submitter_name, dataset_name))
            
            existing = cursor.fetchone()
            if existing:
                print(f"Skipping {dataset_name} for {submitter_name}: Already exists")
                skipped_count += 1
                continue
            
            # Prepare insertion data
            upload_timestamp = datetime.now().isoformat()
            
            # Prepare insertion data with proper schema mapping
            upload_timestamp = datetime.now().isoformat()
            
            # Create evaluation_result structure
            evaluation_result = {
                "status": "success",
                "metrics": metrics,
                "evaluation_details": {
                    "dataset": dataset_name,
                    "imported_from": os.path.basename(json_file_path)
                }
            }
            
            # Insert using your actual database schema
            insert_sql = """
                INSERT INTO output_submissions (
                    dataset_name, submitter_name, submitter_email, 
                    original_filename, upload_timestamp, evaluation_result, 
                    status, description
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            values = (
                dataset_name,
                submitter_name,
                "imported@batch.test",  # submitter_email
                f"imported_from_{os.path.basename(json_file_path)}",  # original_filename
                upload_timestamp,
                json.dumps(evaluation_result),  # evaluation_result as JSON string
                "completed",  # status
                f"Zero-shot evaluation imported from {os.path.basename(json_file_path)}"  # description
            )
            
            cursor.execute(insert_sql, values)
            imported_count += 1
            print(f"✓ Imported: {submitter_name} - {dataset_name}")
            
        except Exception as e:
            print(f"Error importing result {i+1}: {e}")
            skipped_count += 1
            continue
    
    # Commit changes
    try:
        conn.commit()
        print(f"\n✓ Successfully imported {imported_count} results to database")
        if skipped_count > 0:
            print(f"⚠ Skipped {skipped_count} results (duplicates or errors)")
    except Exception as e:
        print(f"Error committing to database: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()
    
    return True

def verify_import(submitter_name, db_path="besesr.db"):
    """Verify the imported results"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Count datasets for the submitter
        cursor.execute("""
            SELECT COUNT(DISTINCT dataset_name) as dataset_count,
                   submitter_name
            FROM output_submissions 
            WHERE submitter_name = ?
        """, (submitter_name,))
        
        result = cursor.fetchone()
        if result:
            dataset_count, name = result
            print(f"\n📊 VERIFICATION RESULTS for {name}:")
            print(f"├─ Datasets: {dataset_count}")
            
            if dataset_count >= 23:
                print(f"✅ {name} should appear on leaderboard!")
            else:
                print(f"⚠ {name} needs {23 - dataset_count} more datasets for leaderboard")
                
            # Show sample evaluation results
            cursor.execute("""
                SELECT dataset_name, evaluation_result 
                FROM output_submissions 
                WHERE submitter_name = ? 
                LIMIT 3
            """, (submitter_name,))
            
            samples = cursor.fetchall()
            if samples:
                print(f"└─ Sample results:")
                for dataset, eval_result in samples:
                    try:
                        result_data = json.loads(eval_result) if eval_result else {}
                        metrics = result_data.get('metrics', {})
                        corr = metrics.get('pearson_correlation', 0)
                        print(f"   • {dataset}: correlation={corr:.3f}")
                    except:
                        print(f"   • {dataset}: imported successfully")
        
        conn.close()
        
    except Exception as e:
        print(f"Error verifying import: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python import_json_results.py <json_file_path> [db_path]")
        print("Example: python import_json_results.py zero_shot_claude-sonnet_1758347251.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    db_file = sys.argv[2] if len(sys.argv) > 2 else "besesr.db"
    
    if not os.path.exists(json_file):
        print(f"Error: JSON file not found: {json_file}")
        sys.exit(1)
    
    if not os.path.exists(db_file):
        print(f"Error: Database file not found: {db_file}")
        sys.exit(1)
    
    print("🔄 Starting JSON import to BESESR database...")
    print(f"📁 JSON file: {json_file}")
    print(f"🗄️ Database: {db_file}")
    
    success = import_json_to_database(json_file, db_file)
    
    if success:
        # Try to extract submitter name from JSON to verify
        try:
            with open(json_file, 'r') as f:
                sample_result = json.load(f)[0]
                submitter_name = sample_result.get('submitter_name') or sample_result.get('model', 'claude-sonnet')
                verify_import(submitter_name, db_file)
        except:
            print("✅ Import completed! Check your leaderboard.")
        
        print(f"\n🎯 Next steps:")
        print(f"1. Restart your BESESR server")
        print(f"2. Check leaderboard at: http://localhost:8000/leaderboard.html")
        print(f"3. Verify with: sqlite3 {db_file} \"SELECT submitter_name, COUNT(*) FROM output_submissions GROUP BY submitter_name;\"")
    else:
        print("❌ Import failed!")
        sys.exit(1)