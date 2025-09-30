# app/config/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import os
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()
DB_TYPE = os.getenv("DB_TYPE", "create-on-boot")


def build_db_url(db_type):
    """Build database URL with authentication credentials from environment variables"""
    if db_type == "local-postgres":
        user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD", "postgres")
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        database = os.getenv("DB_NAME", "sgrades")
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

    elif db_type == "local-mysql":
        user = os.getenv("DB_USER", "root")
        password = os.getenv("DB_PASSWORD", "root")
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "3306")
        database = os.getenv("DB_NAME", "sgrades")
        return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"

    elif db_type == "create-on-boot":
        db_path = os.getenv("DB_PATH", "./sgrades.db")
        return f"sqlite:///{db_path}"

    elif db_type in ["global-postgres", "global-mysql"]:
        # For global databases, require full DATABASE_URL or build from components
        database_url = os.getenv("DATABASE_URL", "")
        if database_url:
            return database_url

        # If DATABASE_URL not provided, try building from components
        user = os.getenv("DB_USER", "")
        password = os.getenv("DB_PASSWORD", "")
        host = os.getenv("DB_HOST", "")
        port = os.getenv("DB_PORT", "")
        database = os.getenv("DB_NAME", "")

        if all([user, password, host, port, database]):
            if db_type == "global-postgres":
                return f"postgresql://{user}:{password}@{host}:{port}/{database}"
            else:
                return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"

        return ""

    return ""


DB_CONFIGS = {
    "local-postgres": {
        "pool_pre_ping": True,
        "pool_recycle": 3600,
        "pool_size": 10,
        "max_overflow": 20,
        "echo": False,
    },
    "local-mysql": {
        "pool_pre_ping": True,
        "pool_recycle": 3600,
        "pool_size": 10,
        "max_overflow": 20,
        "echo": False,
    },
    "create-on-boot": {
        "connect_args": {"check_same_thread": False},
        "poolclass": StaticPool,
        "echo": False,
    },
    "global-postgres": {
        "pool_pre_ping": True,
        "pool_recycle": 3600,
        "pool_size": 10,
        "max_overflow": 20,
        "echo": False,
    },
    "global-mysql": {
        "pool_pre_ping": True,
        "pool_recycle": 3600,
        "pool_size": 10,
        "max_overflow": 20,
        "echo": False,
    },
}

# Validate DB_TYPE
if DB_TYPE not in DB_CONFIGS:
    raise ValueError(f"Invalid DB_TYPE: {DB_TYPE}. " f"Valid options: {', '.join(DB_CONFIGS.keys())}")

# Get database configuration
db_config = DB_CONFIGS[DB_TYPE]
DATABASE_URL = build_db_url(DB_TYPE)

# Validate DATABASE_URL for global databases
if DB_TYPE in ["global-postgres", "global-mysql"] and not DATABASE_URL:
    raise ValueError(f"DATABASE_URL environment variable is required when using DB_TYPE={DB_TYPE}")

print(f"Using database type: {DB_TYPE}")
print(f"Database URL: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL}")

# Create engine with database-specific settings
engine = create_engine(DATABASE_URL, **db_config)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_database():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session():
    """Context manager for database sessions"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_database():
    """Initialize database tables"""
    from app.models.database import Base

    print("🗄️ Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")


def reset_database():
    """Reset database (drop and recreate all tables)"""
    from app.models.database import Base

    print("🗑️ Dropping all database tables...")
    Base.metadata.drop_all(bind=engine)
    print("🗄️ Creating fresh database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database reset completed!")
