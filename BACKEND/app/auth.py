"""
Authentication Module for Kolrose Policy Assistant
====================================================
Handles JWT token creation, validation, user authentication, and RBAC.
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from enum import Enum

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# ============================================================================
# CONFIGURATION
# ============================================================================

SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "kolrose-secret-key-change-in-production-2024")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# FIXED: Use bcrypt with explicit backend detection and fallback
try:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__default_rounds=12)
    # Test if bcrypt works
    pwd_context.hash("test")
except (ValueError, Exception) as e:
    # Fallback to sha256_crypt if bcrypt has issues
    print(f"⚠️ bcrypt unavailable ({e}), using sha256_crypt fallback")
    pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# ============================================================================
# PASSWORD & TOKEN FUNCTIONS (MUST BE BEFORE init_users)
# ============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password - truncate to 72 bytes for bcrypt compatibility"""
    if len(password.encode('utf-8')) > 72:
        password = password[:72]
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt


# ============================================================================
# ROLE DEFINITIONS (RBAC)
# ============================================================================

class UserRole(str, Enum):
    """User roles with hierarchical permissions"""
    ADMIN = "admin"
    HR_MANAGER = "hr_manager"
    MANAGER = "manager"
    EMPLOYEE = "employee"
    VIEWER = "viewer"


# Role hierarchy: higher roles inherit lower role permissions
ROLE_HIERARCHY = {
    UserRole.ADMIN: [UserRole.ADMIN, UserRole.HR_MANAGER, UserRole.MANAGER, UserRole.EMPLOYEE, UserRole.VIEWER],
    UserRole.HR_MANAGER: [UserRole.HR_MANAGER, UserRole.MANAGER, UserRole.EMPLOYEE, UserRole.VIEWER],
    UserRole.MANAGER: [UserRole.MANAGER, UserRole.EMPLOYEE, UserRole.VIEWER],
    UserRole.EMPLOYEE: [UserRole.EMPLOYEE, UserRole.VIEWER],
    UserRole.VIEWER: [UserRole.VIEWER],
}

# ============================================================================
# PERMISSION DEFINITIONS
# ============================================================================

class Permission(str, Enum):
    """Fine-grained permissions"""
    VIEW_POLICIES = "view_policies"
    SEARCH_POLICIES = "search_policies"
    ASK_QUESTIONS = "ask_questions"
    MANAGE_USERS = "manage_users"
    VIEW_USERS = "view_users"
    MANAGE_SYSTEM = "manage_system"
    VIEW_HR_POLICIES = "view_hr_policies"
    MANAGE_HR_POLICIES = "manage_hr_policies"
    VIEW_SENSITIVE_DATA = "view_sensitive_data"
    VIEW_TEAM_DATA = "view_team_data"
    GENERATE_REPORTS = "generate_reports"


# Role-to-permission mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.VIEW_POLICIES, Permission.SEARCH_POLICIES, Permission.ASK_QUESTIONS,
        Permission.MANAGE_USERS, Permission.VIEW_USERS, Permission.MANAGE_SYSTEM,
        Permission.VIEW_HR_POLICIES, Permission.MANAGE_HR_POLICIES,
        Permission.VIEW_SENSITIVE_DATA, Permission.VIEW_TEAM_DATA, Permission.GENERATE_REPORTS,
    ],
    UserRole.HR_MANAGER: [
        Permission.VIEW_POLICIES, Permission.SEARCH_POLICIES, Permission.ASK_QUESTIONS,
        Permission.VIEW_USERS, Permission.VIEW_HR_POLICIES, Permission.MANAGE_HR_POLICIES,
        Permission.VIEW_TEAM_DATA, Permission.GENERATE_REPORTS,
    ],
    UserRole.MANAGER: [
        Permission.VIEW_POLICIES, Permission.SEARCH_POLICIES, Permission.ASK_QUESTIONS,
        Permission.VIEW_TEAM_DATA, Permission.GENERATE_REPORTS,
    ],
    UserRole.EMPLOYEE: [
        Permission.VIEW_POLICIES, Permission.SEARCH_POLICIES, Permission.ASK_QUESTIONS,
    ],
    UserRole.VIEWER: [
        Permission.VIEW_POLICIES, Permission.SEARCH_POLICIES,
    ],
}

# ============================================================================
# DEPARTMENT POLICY ACCESS
# ============================================================================

DEPARTMENT_POLICY_ACCESS = {
    "Human Resources": ["KOL-HR", "KOL-ADMIN"],
    "Information Technology": ["KOL-IT", "KOL-ADMIN"],
    "Finance": ["KOL-FIN", "KOL-ADMIN"],
    "Administration": ["KOL-ADMIN", "KOL-HR", "KOL-FIN", "KOL-IT"],
    "Management": ["KOL-HR", "KOL-IT", "KOL-FIN", "KOL-ADMIN"],
}

# ============================================================================
# USER MODEL
# ============================================================================

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    department: Optional[str] = None
    role: UserRole = UserRole.EMPLOYEE


class UserInDB(User):
    hashed_password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: UserRole
    department: Optional[str] = None
    permissions: List[str] = []


class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[UserRole] = None
    department: Optional[str] = None


class UserCreate(BaseModel):
    username: str
    password: str
    email: str
    full_name: str
    department: str
    role: UserRole = UserRole.EMPLOYEE


# ============================================================================
# USER DATABASE (In-Memory - Replace with real DB in production)
# ============================================================================

USERS_DB: Dict[str, UserInDB] = {}


def init_users():
    """Initialize default users with hashed passwords and RBAC roles"""
    global USERS_DB
    
    if not USERS_DB:
        default_users = [
            UserInDB(
                username="admin",
                email="admin@kolroselimited.com.ng",
                full_name="System Administrator",
                department="Administration",
                role=UserRole.ADMIN,
                hashed_password=get_password_hash("Admin@Kolrose2024"),
            ),
            UserInDB(
                username="hr_manager",
                email="hr@kolroselimited.com.ng",
                full_name="HR Manager",
                department="Human Resources",
                role=UserRole.HR_MANAGER,
                hashed_password=get_password_hash("HR@Kolrose2024"),
            ),
            UserInDB(
                username="manager",
                email="manager@kolroselimited.com.ng",
                full_name="Department Manager",
                department="Management",
                role=UserRole.MANAGER,
                hashed_password=get_password_hash("Manager@Kolrose2024"),
            ),
            UserInDB(
                username="employee",
                email="employee@kolroselimited.com.ng",
                full_name="John Doe",
                department="Human Resources",
                role=UserRole.EMPLOYEE,
                hashed_password=get_password_hash("Employee@Kolrose2024"),
            ),
            UserInDB(
                username="it_support",
                email="it@kolroselimited.com.ng",
                full_name="IT Support",
                department="Information Technology",
                role=UserRole.EMPLOYEE,
                hashed_password=get_password_hash("IT@Kolrose2024"),
            ),
            UserInDB(
                username="finance_user",
                email="finance@kolroselimited.com.ng",
                full_name="Finance Officer",
                department="Finance",
                role=UserRole.EMPLOYEE,
                hashed_password=get_password_hash("Finance@Kolrose2024"),
            ),
            UserInDB(
                username="viewer",
                email="viewer@kolroselimited.com.ng",
                full_name="Guest Viewer",
                department="Administration",
                role=UserRole.VIEWER,
                hashed_password=get_password_hash("Viewer@Kolrose2024"),
            ),
        ]
        
        for user in default_users:
            USERS_DB[user.username] = user


# Initialize users on module load
init_users()


# ============================================================================
# USER LOOKUP & AUTHENTICATION
# ============================================================================

def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database"""
    return USERS_DB.get(username)


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate a user by username and password"""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


# ============================================================================
# RBAC HELPER FUNCTIONS
# ============================================================================

def get_user_permissions(role: UserRole) -> List[str]:
    """Get list of permission strings for a role"""
    return [p.value for p in ROLE_PERMISSIONS.get(role, [])]


def has_permission(user: User, permission: Permission) -> bool:
    """Check if a user has a specific permission"""
    user_permissions = ROLE_PERMISSIONS.get(user.role, [])
    return permission in user_permissions


def can_access_department(user: User, department: str) -> bool:
    """Check if user's department allows access to specific policy categories"""
    if user.role == UserRole.ADMIN:
        return True
    if user.department not in DEPARTMENT_POLICY_ACCESS:
        return False
    allowed_categories = DEPARTMENT_POLICY_ACCESS[user.department]
    return department in allowed_categories


# ============================================================================
# DEPENDENCY FUNCTIONS (for FastAPI route protection)
# ============================================================================

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Validate JWT token and return current user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role_str: str = payload.get("role", "employee")
        
        if username is None:
            raise credentials_exception
        
        role = UserRole(role_str) if role_str in [r.value for r in UserRole] else UserRole.EMPLOYEE
        
    except JWTError:
        raise credentials_exception
    
    user = get_user(username)
    if user is None:
        raise credentials_exception
    
    return user


# ============================================================================
# ROLE-BASED ACCESS CONTROL DEPENDENCIES
# ============================================================================

class RoleChecker:
    """Dependency for role-based access control"""
    
    def __init__(self, allowed_roles: List[UserRole]):
        self.allowed_roles = allowed_roles
    
    def __call__(self, current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {[r.value for r in self.allowed_roles]}"
            )
        return current_user


class PermissionChecker:
    """Dependency for permission-based access control"""
    
    def __init__(self, required_permissions: List[Permission]):
        self.required_permissions = required_permissions
    
    def __call__(self, current_user: User = Depends(get_current_user)) -> User:
        user_permissions = ROLE_PERMISSIONS.get(current_user.role, [])
        for perm in self.required_permissions:
            if perm not in user_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied. Required permission: {perm.value}"
                )
        return current_user


# Convenience dependencies
require_admin = Depends(RoleChecker([UserRole.ADMIN]))
require_hr = Depends(RoleChecker([UserRole.ADMIN, UserRole.HR_MANAGER]))
require_manager = Depends(RoleChecker([UserRole.ADMIN, UserRole.HR_MANAGER, UserRole.MANAGER]))
require_employee = Depends(RoleChecker([UserRole.ADMIN, UserRole.HR_MANAGER, UserRole.MANAGER, UserRole.EMPLOYEE]))
require_authenticated = Depends(get_current_user)

require_manage_users = Depends(PermissionChecker([Permission.MANAGE_USERS]))
require_view_sensitive = Depends(PermissionChecker([Permission.VIEW_SENSITIVE_DATA]))
require_generate_reports = Depends(PermissionChecker([Permission.GENERATE_REPORTS]))


# ============================================================================
# USER MANAGEMENT FUNCTIONS
# ============================================================================

def create_user(user_data: UserCreate) -> UserInDB:
    """Create a new user with RBAC role"""
    if user_data.username in USERS_DB:
        raise ValueError(f"User '{user_data.username}' already exists")
    
    user = UserInDB(
        username=user_data.username,
        email=user_data.email,
        full_name=user_data.full_name,
        department=user_data.department,
        role=user_data.role,
        hashed_password=get_password_hash(user_data.password),
    )
    
    USERS_DB[user_data.username] = user
    return user


def list_users() -> List[dict]:
    """List all users with their roles and permissions"""
    return [
        {
            "username": u.username,
            "email": u.email,
            "full_name": u.full_name,
            "department": u.department,
            "role": u.role.value,
            "permissions": get_user_permissions(u.role),
        }
        for u in USERS_DB.values()
    ]


def update_user_role(username: str, new_role: UserRole) -> bool:
    """Update a user's role"""
    if username not in USERS_DB:
        return False
    USERS_DB[username].role = new_role
    return True


def delete_user(username: str) -> bool:
    """Delete a user"""
    if username in USERS_DB:
        del USERS_DB[username]
        return True
    return False