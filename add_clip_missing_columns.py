from src.shared.core.database import engine
from sqlalchemy import text

def add_missing_columns():
    columns = [
        ("thumbnail_url", "TEXT"),
        ("start_time_sec", "FLOAT"),
        ("end_time_sec", "FLOAT"),
        ("duration_sec", "FLOAT")
    ]
    try:
        with engine.connect() as conn:
            for col_name, col_type in columns:
                try:
                    conn.execute(text(f"ALTER TABLE clips ADD COLUMN IF NOT EXISTS {col_name} {col_type};"))
                    conn.commit()
                    print(f"Successfully added {col_name} column to clips table.")
                except Exception as e:
                    print(f"Error adding {col_name}: {e}")
    except Exception as e:
        print(f"Error connecting to database: {e}")

if __name__ == "__main__":
    add_missing_columns()
