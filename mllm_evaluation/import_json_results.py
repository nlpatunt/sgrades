#!/usr/bin/env python3
"""
Direct database import for Claude Haiku evaluation results
"""
import json
import sqlite3
from datetime import datetime
import os

def import_claude_haiku_results():
    """Import Claude Haiku results directly to database"""
    
    json_file = "zero-shot-claude-haiku_test_full_1758942478.json"
    db_path = "../besesr.db"
    
    # Read the JSON file
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        print(f"Loaded JSON data from {json_file}")
        print(f"Model: {data['model']}")
        print(f"Datasets tested: {data['datasets_tested']}")
        print(f"Results count: {len(data['results'])}")
    except Exception as e:
        print(f"Error reading JSON: {e}")
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
    cursor.execute("PRAGMA table_info(output_submissions);")
    columns = [row[1] for row in cursor.fetchall()]
    print(f"Available columns: {columns}")
    
    # Extract results
    results = data['results']
    submitter_name = "zero-shot-claude-haiku_full"  # Clean name for leaderboard
    
    imported_count = 0
    skipped_count = 0
    
    for result in results:
        try:
            dataset_name = result['dataset']
            metrics = result['metrics']
            
            # Check if already exists
            cursor.execute("""
                SELECT id FROM output_submissions 
                WHERE submitter_name = ? AND dataset_name = ?
            """, (submitter_name, dataset_name))
            
            if cursor.fetchone():
                print(f"Skipping {dataset_name}: Already exists")
                skipped_count += 1
                continue
            
            # Create evaluation result structure matching existing format
            evaluation_result = {
                'real_evaluation': {
                    'status': 'success',
                    'metrics': {
                        'quadratic_weighted_kappa': float(metrics.get('quadratic_weighted_kappa', 0)),
                        'pearson_correlation': float(metrics.get('pearson_correlation', 0)),
                        'mean_absolute_error': float(metrics.get('mean_absolute_error', 0)),
                        'mean_squared_error': float(metrics.get('mean_squared_error', 0)),
                        'root_mean_squared_error': float(metrics.get('root_mean_squared_error', 0)),
                        'f1_score': float(metrics.get('f1_score', 0)),
                        'precision': float(metrics.get('precision', 0)),
                        'recall': float(metrics.get('recall', 0)),
                        'accuracy': float(metrics.get('accuracy', 0))
                    },
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'file_hash_at_evaluation': 'imported_claude_haiku'
                },
                'evaluation_timestamp': datetime.now().isoformat(),
                'file_hash_at_evaluation': 'imported_claude_haiku'
            }
            
            # Insert with exact same format as working entries
            cursor.execute("""
                INSERT INTO output_submissions 
                (dataset_name, submitter_name, submitter_email, 
                 evaluation_result, status, description, 
                 upload_timestamp, stored_file_path, original_filename)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                dataset_name,
                submitter_name,
                'claude.haiku@test.com',
                json.dumps(evaluation_result),
                'completed',
                'Zero-shot Claude Haiku evaluation',
                datetime.now().isoformat(),
                f'claude_haiku_{dataset_name}.csv',
                f'claude_haiku_{dataset_name}.csv'
            ))
            
            imported_count += 1
            print(f"✓ Imported: {dataset_name}")
            
        except Exception as e:
            print(f"Error importing {result.get('dataset', 'unknown')}: {e}")
            skipped_count += 1
            continue
    
    # Commit changes
    try:
        conn.commit()
        print(f"\n✓ Successfully imported {imported_count} results")
        print(f"⚠ Skipped {skipped_count} results")
        
        # Verify the import
        cursor.execute("""
            SELECT COUNT(DISTINCT dataset_name) 
            FROM output_submissions 
            WHERE submitter_name = ?
        """, (submitter_name,))
        
        count = cursor.fetchone()[0]
        print(f"📊 {submitter_name} now has {count} datasets in database")
        
        if count >= 20:
            print(f"🏆 {submitter_name} should appear on leaderboard!")
        else:
            print(f"⚠ Needs {20 - count} more datasets for leaderboard")
            
    except Exception as e:
        print(f"Error committing: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()
    
    return True

def verify_import():
    """Verify the imported results"""
    try:
        conn = sqlite3.connect("../besesr.db")
        cursor = conn.cursor()
        
        # Check all models with counts
        cursor.execute("""
            SELECT submitter_name, COUNT(DISTINCT dataset_name) as datasets 
            FROM output_submissions 
            GROUP BY submitter_name 
            ORDER BY datasets DESC
        """)
        
        results = cursor.fetchall()
        print("\n📊 Current leaderboard status:")
        print("=" * 50)
        for name, count in results:
            status = "🏆 ON LEADERBOARD" if count >= 20 else f"⚠ NEEDS {20-count} MORE"
            print(f"{name}: {count} datasets {status}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error verifying: {e}")

if __name__ == "__main__":
    print("🔄 Importing Claude Haiku results to BESESR database...")
    
    if import_claude_haiku_results():
        verify_import()
        print(f"\n🎯 Next steps:")
        print(f"1. Restart your BESESR server")
        print(f"2. Check leaderboard at: http://localhost:8000/leaderboard.html")
        print(f"3. Look for 'zero-shot-claude-haiku_full' on the leaderboard")
    else:
        print("❌ Import failed!")