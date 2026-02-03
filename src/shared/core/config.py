from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings"""

    # App Info
    PROJECT_NAME: str = Field(default="FastAPI App", description="Project name")
    RELOAD: bool = Field(default=True, description="Reload the server on code changes")

    # Server
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")

    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FILE: Optional[str] = Field(
        default=None, nullable=True, description="Log file path"
    )

    #api keys
    ASSEMBLYAI_API_KEY: str = Field(description="AssemblyAI API key")
    OPENAI_API_KEY: str = Field(description="OpenAI API key")

    # CORS
    ALLOWED_ORIGINS: List[str] = Field(default=["*"], description="Allowed origins for CORS")
    
    # Database
    DATABASE_URL: str = Field(description="PostgreSQL database URL")
    
    # Azure Storage
    AZURE_STORAGE_ACCOUNT_NAME: str = Field(description="Azure Storage account name")
    AZURE_STORAGE_ACCOUNT_KEY: str = Field(description="Azure Storage account key")
    AZURE_STORAGE_CONTAINER_NAME: str = Field(description="Azure Storage container name")
    AZURE_STORAGE_CONNECTION_STRING: str = Field(description="Azure Storage connection string")
    
    # Azure Storage - Thumbnail Generator
    THUMBNAIL_STORAGE_CONNECTION_STRING: str = Field(
        description="Azure Storage connection string for thumbnail generator"
    )
    THUMBNAIL_STORAGE_ACCOUNT_NAME: str = Field(description="Thumbnail storage account name")
    
    # Azure Queue Storage
    AZURE_QUEUE_NAME: str = Field(description="Azure Queue name for video processing")
    THUMBNAIL_QUEUE_NAME: str = Field(default="generatethumbnailqueue", description="Queue name for thumbnail generation")
    
    # JWT Authentication
    JWT_SECRET_KEY: str = Field(description="Secret key for JWT token signing")
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=15, description="Access token expiry in minutes")
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=30, description="Refresh token expiry in days")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()