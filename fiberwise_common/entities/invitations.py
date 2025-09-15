"""
Invitation entities and schemas for the Fiberwise platform.
These models handle user invitations to organizations and teams.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field


class InvitationRequest(BaseModel):
    """Model for creating a new invitation"""
    email: EmailStr = Field(..., description="Email address of the person to invite")
    role: str = Field(default="member", description="Role to assign to the invited user")
    team_id: Optional[int] = Field(None, description="Optional team ID to add the user to")
    message: Optional[str] = Field(None, description="Optional message to include in the invitation")


class InvitationResponse(BaseModel):
    """Response model for invitation information"""
    id: int = Field(..., description="Invitation ID")
    uuid: str = Field(..., description="Invitation UUID")
    email: str = Field(..., description="Invited email address")
    role: str = Field(..., description="Role the user will be assigned")
    team_id: Optional[int] = Field(None, description="Team ID if invited to a specific team")
    status: str = Field(..., description="Invitation status")
    message: Optional[str] = Field(None, description="Invitation message")
    expires_at: str = Field(..., description="Invitation expiration timestamp")
    created_at: str = Field(..., description="Invitation creation timestamp")


class InvitationUpdate(BaseModel):
    """Model for updating invitation details"""
    role: Optional[str] = Field(None, description="Updated role")
    team_id: Optional[int] = Field(None, description="Updated team ID")
    message: Optional[str] = Field(None, description="Updated invitation message")
    expires_at: Optional[datetime] = Field(None, description="Updated expiration timestamp")


class InvitationAcceptRequest(BaseModel):
    """Model for accepting an invitation"""
    invitation_token: str = Field(..., description="Invitation token from email/link")
    password: Optional[str] = Field(None, description="Password for new user account creation")
    first_name: Optional[str] = Field(None, description="First name for new user")
    last_name: Optional[str] = Field(None, description="Last name for new user")


class InvitationValidationResponse(BaseModel):
    """Response model for invitation validation"""
    is_valid: bool = Field(..., description="Whether the invitation is valid")
    invitation_id: Optional[int] = Field(None, description="Invitation ID if valid")
    email: Optional[str] = Field(None, description="Invited email if valid")
    role: Optional[str] = Field(None, description="Role if valid")
    organization_name: Optional[str] = Field(None, description="Organization name if valid")
    team_name: Optional[str] = Field(None, description="Team name if applicable")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp if valid")
    error_message: Optional[str] = Field(None, description="Error message if invalid")


class BulkInvitationRequest(BaseModel):
    """Model for sending multiple invitations at once"""
    invitations: list[InvitationRequest] = Field(..., description="List of invitations to send")
    send_email: bool = Field(default=True, description="Whether to send invitation emails")


class BulkInvitationResponse(BaseModel):
    """Response model for bulk invitation results"""
    successful_invitations: list[InvitationResponse] = Field(..., description="Successfully created invitations")
    failed_invitations: list[dict] = Field(..., description="Failed invitations with error details")
    total_sent: int = Field(..., description="Total number of invitations processed")
    successful_count: int = Field(..., description="Number of successful invitations")
    failed_count: int = Field(..., description="Number of failed invitations")


class InvitationListItem(BaseModel):
    """Simplified invitation model for list responses"""
    id: int = Field(..., description="Invitation ID")
    email: str = Field(..., description="Invited email")
    role: str = Field(..., description="Assigned role")
    status: str = Field(..., description="Invitation status")
    created_at: datetime = Field(..., description="Creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    invited_by_name: Optional[str] = Field(None, description="Name of person who sent invitation")


class InvitationStats(BaseModel):
    """Model for invitation statistics"""
    total_invitations: int = Field(..., description="Total invitations sent")
    pending_invitations: int = Field(..., description="Pending invitations")
    accepted_invitations: int = Field(..., description="Accepted invitations")
    expired_invitations: int = Field(..., description="Expired invitations")
    declined_invitations: int = Field(..., description="Declined invitations")