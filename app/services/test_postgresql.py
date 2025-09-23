import psycopg2
import urllib.parse
user = "postgres"
password = "Asdpoizxc123!!!"   # raw password, no encoding needed
host = "db.osugaozwiczewgsmlwbt.supabase.co"
port = 5432
database = "postgres"
# URL-encode the password
encoded_password = urllib.parse.quote_plus(password)
url = f"postgresql://{user}:{encoded_password}@{host}:{port}/{database}?sslmode=require"
try:
   conn = psycopg2.connect(url)
   cur = conn.cursor()
   cur.execute("SELECT version();")
   version = cur.fetchone()
   print("✅ Connected! PostgreSQL version:", version[0])
   cur.close()
   conn.close()
except Exception as e:
   print("❌ Connection failed:", e)