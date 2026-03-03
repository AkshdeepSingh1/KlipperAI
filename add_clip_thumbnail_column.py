from src.shared.core.database import engine
from sqlalchemy import text

def add_thumbnail_column():
    try:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE clips ADD COLUMN IF NOT EXISTS thumbnail_url TEXT;"))
            conn.commit()
            print("Successfully added thumbnail_url column to clips table.")
    except Exception as e:
        print(f"Error adding column: {e}")

if __name__ == "__main__":
    add_thumbnail_column()
