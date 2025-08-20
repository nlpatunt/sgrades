#app/api/config/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import os
from contextlib import contextmanager

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./besesr.db")

# Create engine with SQLite-specific settings
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False  # Set to True for SQL debugging
    )
else:
    engine = create_engine(DATABASE_URL)

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