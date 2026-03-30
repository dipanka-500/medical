from .service import AuthService
from .dependencies import get_current_user, require_role
from .schemas import (
    RegisterRequest, LoginRequest, LoginResponse, TokenRefreshRequest,
    TokenRefreshResponse, Enable2FAResponse, Verify2FARequest,
)


def require_verified():
    """Dependency: require the current user to have a verified email."""
    from fastapi import Depends, HTTPException, status
    from .dependencies import get_current_user as _get_user
    from db.models import User

    async def _check(user: User = Depends(_get_user)) -> User:
        if not user.is_verified:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Email verification required. Please verify your email.",
            )
        return user

    return _check
