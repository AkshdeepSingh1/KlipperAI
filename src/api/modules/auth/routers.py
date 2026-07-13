from fastapi import APIRouter, Depends, HTTPException, status, Response, Cookie, Request
from sqlalchemy.orm import Session
import uuid
from src.shared.core.database import get_db
from src.shared.core.config import settings
from src.shared.core.logger import get_logger
from .schemas import (
    RegisterRequest, 
    LoginRequest, 
    AuthResponse, 
    UserResponse,
    ErrorResponse
)
from .service import AuthService

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])


def _cookie_security_kwargs() -> dict:
    """Shared security attributes for all auth cookies.

    Driven by settings so the same code works for local dev (same-site over HTTP)
    and cross-origin production (SameSite=None over HTTPS). delete_cookie must be
    given the same secure/samesite values or the browser won't match and clear it.
    """
    return {
        "secure": settings.COOKIE_SECURE,
        "samesite": settings.COOKIE_SAMESITE,
    }


@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request - validation error or user already exists"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Register a new user",
    description="Create a new user account and return authentication tokens"
)
async def register(
    request: RegisterRequest,
    response: Response,
    db: Session = Depends(get_db)
):
    """
    Register a new user with email and password.
    
    - **name**: User's full name (optional)
    - **email**: Valid email address (required, must be unique)
    - **password**: Password with minimum 8 characters (required)
    
    Sets HTTP-only cookies for auth_token, refresh_token, and device_id.
    """
    try:
        logger.info(f"Registration attempt for email: {request.email}")
        
        # Create user
        user = AuthService.create_user(
            db=db,
            name=request.name,
            email=request.email,
            password=request.password
        )
        
        # Generate device_id
        device_id = str(uuid.uuid4())
        
        # Generate tokens
        access_token, refresh_token = AuthService.generate_tokens(user.id)
        
        # Create auth session
        auth_session = AuthService.create_auth_session(
            db=db,
            user_id=user.id,
            refresh_token=refresh_token,
            device_id=device_id
        )
        
        logger.info(f"User registered successfully: {user.email} (ID: {user.id})")
        
        # Set HTTP-only cookies
        response.set_cookie(
            key="auth_token",
            value=access_token,
            httponly=True,
            **_cookie_security_kwargs(),
            max_age=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        response.set_cookie(
            key="refresh_token",
            value=refresh_token,
            httponly=True,
            **_cookie_security_kwargs(),
            max_age=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
        )
        response.set_cookie(
            key="device_id",
            value=device_id,
            httponly=True,
            **_cookie_security_kwargs(),
            max_age=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
        )
        
        return AuthResponse(
            message="User registered successfully",
            user=UserResponse.model_validate(user)
        )
        
    except ValueError as e:
        logger.warning(f"Registration failed for {request.email}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Registration error for {request.email}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during registration"
        )


@router.post(
    "/login",
    response_model=AuthResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid credentials"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Login user",
    description="Authenticate user with email and password and return tokens"
)
async def login(
    request: LoginRequest,
    response: Response,
    db: Session = Depends(get_db)
):
    """
    Login with email and password.
    
    - **email**: User's email address
    - **password**: User's password
    
    Sets HTTP-only cookies for auth_token, refresh_token, and device_id.
    """
    try:
        logger.info(f"Login attempt for email: {request.email}")
        
        # Authenticate user
        user = AuthService.authenticate_user(
            db=db,
            email=request.email,
            password=request.password
        )
        
        if not user:
            logger.warning(f"Failed login attempt for email: {request.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Generate device_id
        device_id = str(uuid.uuid4())
        
        # Generate tokens
        access_token, refresh_token = AuthService.generate_tokens(user.id)
        
        # Create auth session (this will revoke any existing session for the device)
        auth_session = AuthService.create_auth_session(
            db=db,
            user_id=user.id,
            refresh_token=refresh_token,
            device_id=device_id
        )
        
        logger.info(f"User logged in successfully: {user.email} (ID: {user.id})")
        
        # Set HTTP-only cookies
        response.set_cookie(
            key="auth_token",
            value=access_token,
            httponly=True,
            **_cookie_security_kwargs(),
            max_age=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        response.set_cookie(
            key="refresh_token",
            value=refresh_token,
            httponly=True,
            **_cookie_security_kwargs(),
            max_age=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
        )
        response.set_cookie(
            key="device_id",
            value=device_id,
            httponly=True,
            **_cookie_security_kwargs(),
            max_age=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
        )
        
        return AuthResponse(
            message="Login successful",
            user=UserResponse.model_validate(user)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error for {request.email}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during login"
        )


@router.post(
    "/refresh",
    response_model=AuthResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid or expired refresh token"},
        403: {"model": ErrorResponse, "description": "Token reuse detected - all sessions revoked"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Refresh access token",
    description="Generate new access and refresh tokens using a valid refresh token from cookies"
)
async def refresh_token(
    response: Response,
    db: Session = Depends(get_db),
    refresh_token: str = Cookie(None),
    device_id: str = Cookie(None)
):
    """
    Refresh access token using refresh token from cookies.
    
    Reads refresh_token and device_id from HTTP-only cookies.
    
    **Security Features:**
    - Refresh token rotation (old token is invalidated, new one issued)
    - Token reuse detection (if revoked token is used, all user sessions are revoked)
    - Validates token against database hash
    - Updates refresh token hash in database
    """
    try:
        logger.info("Refresh token request received")
        
        if not refresh_token or not device_id:
            logger.warning("Missing refresh_token or device_id cookie")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authentication cookies"
            )
        
        # Decode and verify the refresh token JWT
        payload = AuthService.verify_token(refresh_token, "refresh")
        if not payload:
            logger.warning("Invalid or expired refresh token JWT")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token"
            )
        
        user_id = payload.get("user_id")
        if not user_id:
            logger.warning("Missing user_id in refresh token payload")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        # Hash the refresh token to compare with database
        refresh_token_hash = AuthService.hash_refresh_token(refresh_token)
        
        # Query the database for the session
        from sqlalchemy import and_
        from src.shared.models import AuthSession
        from datetime import datetime, timezone, timedelta
        
        session = db.query(AuthSession).filter(
            and_(
                AuthSession.user_id == user_id,
                AuthSession.device_id == device_id,
                AuthSession.refresh_token_hash == refresh_token_hash,
                AuthSession.expires_at > datetime.now(timezone.utc)
            )
        ).first()
        
        if not session:
            logger.warning(f"No matching session found for user {user_id} and device {device_id}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token"
            )
        
        # Check if session is already revoked (token reuse detection)
        if session.revoked:
            logger.critical(f"Token reuse detected for user {user_id}. Revoking all sessions for device {device_id}.")
            
            # Security breach - revoke all sessions for this user and device
            AuthService.revoke_all_user_sessions(db, user_id)
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Token reuse detected. All sessions have been revoked for security."
            )
        
        # Get user information
        user = session.user
        if not user:
            logger.error(f"User not found for session {session.id}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid session"
            )
        
        # Generate new tokens
        new_access_token, new_refresh_token = AuthService.generate_tokens(user.id)
        
        # Update the session with new refresh token hash
        new_refresh_token_hash = AuthService.hash_refresh_token(new_refresh_token)
        session.refresh_token_hash = new_refresh_token_hash
        session.last_used_at = datetime.now(timezone.utc)
        session.expires_at = datetime.now(timezone.utc) + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
        db.commit()
        
        logger.info(f"Tokens refreshed successfully for user: {user.email} (ID: {user.id})")
        
        # Set new HTTP-only cookies
        response.set_cookie(
            key="auth_token",
            value=new_access_token,
            httponly=True,
            **_cookie_security_kwargs(),
            max_age=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        response.set_cookie(
            key="refresh_token",
            value=new_refresh_token,
            httponly=True,
            **_cookie_security_kwargs(),
            max_age=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
        )
        
        return AuthResponse(
            message="Tokens refreshed successfully",
            user=UserResponse.model_validate(user)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Refresh token error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during token refresh"
        )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid refresh token"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Logout user",
    description="Revoke the current refresh token session and clear cookies"
)
async def logout(
    response: Response,
    db: Session = Depends(get_db),
    refresh_token: str = Cookie(None),
    device_id: str = Cookie(None)
):
    """
    Logout user by revoking the refresh token session and clearing cookies.
    
    Reads refresh_token and device_id from HTTP-only cookies.
    """
    try:
        logger.info("Logout request received")
        
        if not refresh_token or not device_id:
            logger.warning("Missing refresh_token or device_id cookie in logout")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authentication cookies"
            )
        
        # Validate and get session
        session = AuthService.validate_refresh_token(db, refresh_token)
        
        if not session:
            logger.warning("Invalid refresh token used in logout")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Revoke the session
        AuthService.revoke_session(db, session)
        
        logger.info(f"User logged out successfully: user_id={session.user_id}")
        
        # Clear all authentication cookies (must match secure/samesite to be cleared)
        response.delete_cookie(key="auth_token", **_cookie_security_kwargs())
        response.delete_cookie(key="refresh_token", **_cookie_security_kwargs())
        response.delete_cookie(key="device_id", **_cookie_security_kwargs())
        
        return None  # 204 No Content
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during logout"
        )
