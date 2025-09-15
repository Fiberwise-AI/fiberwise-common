"""
App Reviews and Ratings Entities
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, Field, validator


class AppReviewCreate(BaseModel):
    """Request model for creating an app review"""
    app_id: UUID
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5 stars")
    title: Optional[str] = Field(None, max_length=200, description="Review title")
    review_text: Optional[str] = Field(None, description="Review content")
    
    @validator('title')
    def validate_title(cls, v):
        if v is not None and len(v.strip()) == 0:
            return None
        return v
    
    @validator('review_text')
    def validate_review_text(cls, v):
        if v is not None and len(v.strip()) == 0:
            return None
        return v


class AppReviewUpdate(BaseModel):
    """Request model for updating an app review"""
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating from 1 to 5 stars")
    title: Optional[str] = Field(None, max_length=200, description="Review title")
    review_text: Optional[str] = Field(None, description="Review content")
    
    @validator('title')
    def validate_title(cls, v):
        if v is not None and len(v.strip()) == 0:
            return None
        return v
    
    @validator('review_text')
    def validate_review_text(cls, v):
        if v is not None and len(v.strip()) == 0:
            return None
        return v


class AppReviewRead(BaseModel):
    """Response model for app reviews"""
    review_id: UUID
    app_id: UUID
    user_id: int
    rating: int
    title: Optional[str]
    review_text: Optional[str]
    is_verified_install: bool
    helpful_count: int
    created_at: datetime
    updated_at: datetime
    
    # User information
    reviewer_name: str
    reviewer_email: Optional[str] = None  # Only shown to app owners/admins
    
    # Current user's interaction with this review
    user_voted_helpful: Optional[bool] = None  # true=helpful, false=not helpful, null=no vote


class AppRatingsSummary(BaseModel):
    """Summary of ratings for an app"""
    app_id: UUID
    total_reviews: int
    average_rating: Optional[float]
    rating_distribution: dict  # {"1": count, "2": count, ...}


class ReviewVoteCreate(BaseModel):
    """Request model for voting on review helpfulness"""
    is_helpful: bool


class ReviewVoteRead(BaseModel):
    """Response model for review votes"""
    vote_id: UUID
    review_id: UUID
    user_id: int
    is_helpful: bool
    created_at: datetime


# Add ratings summary to existing MarketplaceAppRead
class MarketplaceAppWithRatings(BaseModel):
    """Extended marketplace app with ratings information"""
    # All fields from MarketplaceAppRead
    app_id: UUID
    name: str
    description: Optional[str]
    version: str
    marketplace_status: str
    creator_user_id: int
    created_at: datetime
    publisher_name: str
    category: str
    icon: Optional[str]
    screenshots: List[str]
    install_count: int
    is_installed: bool
    install_path: Optional[str]
    
    # Featured app fields
    icon_class: Optional[str] = None
    tags: Optional[str] = None
    stats_json: Optional[str] = None
    install_command: Optional[str] = None
    tutorial_url: Optional[str] = None
    source_url: Optional[str] = None
    featured_tags: Optional[List[str]] = None
    featured_stats: Optional[dict] = None
    
    # Ratings information
    ratings_summary: Optional[AppRatingsSummary] = None
    user_review: Optional[AppReviewRead] = None  # Current user's review if any
    recent_reviews: Optional[List[AppReviewRead]] = None  # Recent reviews for display
