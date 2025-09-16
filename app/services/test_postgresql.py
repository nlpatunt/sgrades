import os
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

load_dotenv()

def test_postgresql_connection():
    DATABASE_URL = os.getenv("DATABASE_URL")
    print(f"Attempting to connect to: {DATABASE_URL[:50]}...")
    
    try:
        print("\n1. Connecting to PostgreSQL...")
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        print("✓ Connection successful!")
        
        print("\n2. Creating test table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_submissions (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(100),
                score DECIMAL(5,3),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        print("✓ Table created successfully!")
        

        print("\n3. Inserting test data...")
        cursor.execute("""
            INSERT INTO test_submissions (model_name, score) 
            VALUES (%s, %s)
        """, ("GPT-4", 0.875))
        
        cursor.execute("""
            INSERT INTO test_submissions (model_name, score) 
            VALUES (%s, %s)
        """, ("Claude-3", 0.823))
        
        conn.commit()
        print("✓ Test data inserted successfully!")
        
        print("\n4. Querying test data...")
        cursor.execute("SELECT * FROM test_submissions ORDER BY score DESC")
        results = cursor.fetchall()
        
        print("Results from PostgreSQL:")
        for row in results:
            print(f"  ID: {row[0]}, Model: {row[1]}, Score: {row[2]}, Created: {row[3]}")

        print("\n5. Cleaning up test table...")
        cursor.execute("DROP TABLE test_submissions")
        conn.commit()
        print("✓ Test table dropped successfully!")
        
 
        cursor.close()
        conn.close()
        print("\n✓ PostgreSQL test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ PostgreSQL connection failed!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        return False

if __name__ == "__main__":
    print("PostgreSQL Connection Test")
    print("=" * 50)
    

    if not os.getenv("DATABASE_URL"):
        print("ERROR: DATABASE_URL not found in environment variables")
        print("Please check your .env file")
        exit(1)
    
    success = test_postgresql_connection()
    
    if success:
        print("\nPostgreSQL is working correctly!")
    else:
        print("\nPostgreSQL connection failed")