"""
Organization and team management entities and schemas for the Fiberwise platform.
These models handle organization CRUD, member management, teams, and permissions.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field
import re


class OrganizationCreate(BaseModel):
    """Model for creating a new organization"""
    name: str = Field(..., min_length=1, max_length=100, description="Organization name")
    display_name: Optional[str] = Field(None, max_length=100, description="Display name for the organization")
    description: Optional[str] = Field(None, max_length=500, description="Organization description")
    website: Optional[str] = Field(None, max_length=200, description="Organization website URL")
    billing_email: Optional[str] = Field(None, max_length=100, description="Billing contact email")


class OrganizationUpdate(BaseModel):
    """Model for updating organization details"""
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="Updated organization name")
    display_name: Optional[str] = Field(None, max_length=100, description="Updated display name")
    description: Optional[str] = Field(None, max_length=500, description="Updated description")
    website: Optional[str] = Field(None, max_length=200, description="Updated website URL")
    billing_email: Optional[str] = Field(None, max_length=100, description="Updated billing email")
    subscription_tier: Optional[str] = Field(None, description="Updated subscription tier")


class OrganizationResponse(BaseModel):
    """Response model for organization information"""
    id: int = Field(..., description="Organization ID")
    name: str = Field(..., description="Organization name")
    display_name: Optional[str] = Field(None, description="Display name")
    description: Optional[str] = Field(None, description="Organization description")
    website: Optional[str] = Field(None, description="Organization website")
    billing_email: Optional[str] = Field(None, description="Billing email")
    subscription_tier: Optional[str] = Field(None, description="Subscription tier")
    member_count: Optional[int] = Field(None, description="Number of members")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class TeamCreate(BaseModel):
    """Model for creating a new team within an organization"""
    name: str = Field(..., min_length=1, max_length=50, description="Team name")
    description: Optional[str] = Field(None, max_length=200, description="Team description")
    color: Optional[str] = Field("#3B82F6", pattern=r"^#[0-9A-Fa-f]{6}$", description="Team color (hex format)")


class TeamUpdate(BaseModel):
    """Model for updating team details"""
    name: Optional[str] = Field(None, min_length=1, max_length=50, description="Updated team name")
    description: Optional[str] = Field(None, max_length=200, description="Updated team description")
    color: Optional[str] = Field(None, pattern=r"^#[0-9A-Fa-f]{6}$", description="Updated team color")


class TeamResponse(BaseModel):
    """Response model for team information"""
    id: int = Field(..., description="Team ID")
    name: str = Field(..., description="Team name")
    description: Optional[str] = Field(None, description="Team description")
    color: str = Field(..., description="Team color")
    organization_id: int = Field(..., description="Parent organization ID")
    member_count: Optional[int] = Field(None, description="Number of team members")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class MemberInvite(BaseModel):
    """Model for inviting a member to an organization"""
    email: str = Field(..., pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", description="Member email address")
    role: str = Field(..., pattern=r"^(owner|admin|member|viewer)$", description="Member role")
    team_id: Optional[int] = Field(None, description="Optional team ID to add member to")
    message: Optional[str] = Field(None, max_length=500, description="Optional invitation message")


class MemberUpdate(BaseModel):
    """Model for updating organization member details"""
    role: str = Field(..., pattern=r"^(owner|admin|member|viewer)$", description="Updated member role")


class MemberResponse(BaseModel):
    """Response model for organization member information"""
    id: int = Field(..., description="Member ID")
    user_id: int = Field(..., description="User ID")
    email: str = Field(..., description="Member email")
    role: str = Field(..., description="Member role")
    status: str = Field(..., description="Member status (active, pending, suspended)")
    organization_id: int = Field(..., description="Organization ID")
    teams: Optional[List[Dict[str, Any]]] = Field(default=[], description="Teams the member belongs to")
    joined_at: Optional[datetime] = Field(None, description="When the member joined")
    invited_at: Optional[datetime] = Field(None, description="When the member was invited")


class TeamMemberAction(BaseModel):
    """Model for team member actions (add/remove from team)"""
    team_id: int = Field(..., description="Team ID")
    role: Optional[str] = Field("member", pattern=r"^(lead|member)$", description="Team role")


class InvitationResponse(BaseModel):
    """Response model for invitation information"""
    id: int = Field(..., description="Invitation ID")
    email: str = Field(..., description="Invited email address")
    role: str = Field(..., description="Invited role")
    organization_id: int = Field(..., description="Organization ID")
    team_id: Optional[int] = Field(None, description="Team ID if invited to specific team")
    status: str = Field(..., description="Invitation status (pending, accepted, expired)")
    invited_by: int = Field(..., description="User ID who sent the invitation")
    created_at: datetime = Field(..., description="Invitation creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Invitation expiration timestamp")


class OrganizationSettings(BaseModel):
    """Model for organization settings and preferences"""
    allow_public_signup: Optional[bool] = Field(None, description="Allow public signup to organization")
    require_email_verification: Optional[bool] = Field(None, description="Require email verification for new members")
    default_member_role: Optional[str] = Field(None, pattern=r"^(member|viewer)$", description="Default role for new members")
    max_members: Optional[int] = Field(None, description="Maximum number of members allowed")
    features_enabled: Optional[List[str]] = Field(None, description="List of enabled features")


class OrganizationStats(BaseModel):
    """Model for organization statistics and metrics"""
    total_members: int = Field(..., description="Total number of members")
    active_members: int = Field(..., description="Number of active members")
    pending_invitations: int = Field(..., description="Number of pending invitations")
    total_teams: int = Field(..., description="Total number of teams")
    apps_count: Optional[int] = Field(None, description="Number of apps in organization")
    agents_count: Optional[int] = Field(None, description="Number of agents in organization")
    created_at: datetime = Field(..., description="Organization creation date")