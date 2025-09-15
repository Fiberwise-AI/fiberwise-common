"""User schemas for the FiberWise platform"""

from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime


class UserBase(BaseModel):
    """Base User schema with common attributes"""
    username: Optional[str] = None
    email: EmailStr
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_active: bool = True
    is_superuser: bool = False


class UserCreate(UserBase):
    """Schema for creating a new user"""
    password: str
    
    model_config = ConfigDict(from_attributes=True)


class User(UserBase):
    """Complete User schema returned from the database"""
    id: int
    uuid: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    organization_id: Optional[int] = None
    settings: Optional[Dict[str, Any]] = None
    profile_image_url: Optional[str] = None
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    timezone: Optional[str] = None
    locale: Optional[str] = None
    is_verified: bool = False
    is_admin: bool = False
    
    model_config = ConfigDict(from_attributes=True)


class UserUpdate(BaseModel):
    """Schema for updating a user"""
    email: Optional[EmailStr] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    settings: Optional[Dict[str, Any]] = None
    profile_image_url: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class UserInDB(User):
    """User schema with hashed password for internal use"""
    hashed_password: str
    
    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    """Authentication token schema"""
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Token payload schema"""
    sub: Optional[str] = None
    exp: Optional[datetime] = None