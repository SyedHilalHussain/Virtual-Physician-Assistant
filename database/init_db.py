"""
Database initialization script.
Run this to create all tables and seed initial data.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import settings
from database.models import Base
from database.crud import create_doctor, create_patient, get_all_doctors, get_all_patients


def init_database():
    """Create all tables and seed initial data."""
    engine = create_engine(settings.DATABASE_URL, echo=True)

    # Create all tables
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully.")

    # Seed initial data
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    try:
        # Add a default doctor if none exist
        doctors = get_all_doctors(db)
        if not doctors:
            print("Seeding default doctor...")
            create_doctor(
                db,
                name="Dr. Sara",
                email="doctor@mhp.com",
                specialty="Cancer Pain Management"
            )
            print("Default doctor created: Dr. Sara (doctor@mhp.com)")

        # Add a sample patient if none exist
        patients = get_all_patients(db)
        if not patients:
            print("Seeding sample patient...")
            create_patient(
                db,
                name="Hilal",
                age=35,
                gender="Male",
                diagnosis="Cancer Pain Management",
                care_type="palliative"
            )
            print("Sample patient created: Hilal")

        print("\nDatabase initialization complete!")
        print(f"Doctors: {len(get_all_doctors(db))}")
        print(f"Patients: {len(get_all_patients(db))}")

    finally:
        db.close()


if __name__ == "__main__":
    init_database()
