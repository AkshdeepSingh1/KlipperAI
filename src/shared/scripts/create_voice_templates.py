from sqlalchemy import text
from src.shared.core.database import engine

sql = """
CREATE TABLE IF NOT EXISTS voice_templates (
    id BIGSERIAL PRIMARY KEY,

    name TEXT NOT NULL,
    category TEXT NOT NULL,
    accent TEXT,
    tone TEXT,

    provider TEXT NOT NULL,                -- e.g. 'elevenlabs', 'azure', 'google'
    provider_voice_id TEXT NOT NULL,       -- voice id from provider

    preview_audio_url TEXT,                -- CDN/Blob URL
    in_house_db_url TEXT,                  -- your internal storage reference

    is_active BOOLEAN DEFAULT TRUE,
    sort_order INT DEFAULT 0,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
"""

def create_table():
    try:
        with engine.connect() as conn:
            conn.execute(text(sql))
            conn.commit()
            print("Table 'voice_templates' created successfully.")
    except Exception as e:
        print(f"Error creating table: {e}")

if __name__ == "__main__":
    create_table()
