"""
Organization Management Service
Handles organization CRUD operations, member management, teams, and permissions.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import uuid
import secrets
import string
from fiberwise_common import DatabaseProvider


class OrganizationService:
    def __init__(self, db: DatabaseProvider):
        self.db = db

    async def create_organization(
        self,
        name: str,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        website: Optional[str] = None,
        billing_email: Optional[str] = None,
        created_by: int = None
    ):
        """Create a new organization"""
        # Generate UUID and slug
        org_uuid = str(uuid.uuid4())
        slug = await self._generate_slug(name)
        
        # Insert organization
        query = """
            INSERT INTO organizations (
                uuid, name, display_name, description, slug, website, 
                billing_email, subscription_tier, is_active, created_by, 
                created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, 'free', 1, $8, 
                datetime('now'), datetime('now')
            )
        """
        
        await self.db.execute(query, 
            org_uuid, name, display_name, description, slug, website, 
            billing_email, created_by
        )
        
        # Get the created organization
        org = await self.db.fetch_one("SELECT * FROM organizations WHERE uuid = $1", org_uuid)
        return org

    async def _generate_slug(self, name: str) -> str:
        """Generate a unique slug from organization name"""
        # Basic slug generation
        slug = name.lower().replace(' ', '-').replace('_', '-')
        # Remove special characters
        slug = ''.join(c for c in slug if c.isalnum() or c == '-')
        
        # Check if slug exists and make unique
        counter = 1
        original_slug = slug
        while True:
            existing = await self.db.fetch_one("SELECT id FROM organizations WHERE slug = $1", slug)
            if not existing:
                break
            slug = f"{original_slug}-{counter}"
            counter += 1
        
        return slug

    async def get_organization(self, org_id: int):
        """Get organization by ID"""
        query = "SELECT * FROM organizations WHERE id = $1 AND (is_active = 1 OR is_active = true OR is_active = 'true')"
        return await self.db.fetch_one(query, org_id)

    async def get_user_organizations(self, user_id: int):
        """Get organizations the user belongs to"""
        query = """
            SELECT o.*, om.role, om.joined_at,
                   (SELECT COUNT(*) FROM organization_members om2 WHERE om2.organization_id = o.id AND om2.status = 'active') as member_count
            FROM organizations o
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = $1 AND om.status = 'active' AND (o.is_active = 1 OR o.is_active = true OR o.is_active = 'true')
            ORDER BY o.name
        """
        return await self.db.fetch_all(query, user_id)

    async def update_organization(self, org_id: int, update_data: Dict[str, Any]):
        """Update organization details"""
        if not update_data:
            return await self.get_organization(org_id)
        
        # Build dynamic update query
        set_clauses = []
        params = [org_id]
        param_index = 2
        
        for key, value in update_data.items():
            if key in ['name', 'display_name', 'description', 'website', 'billing_email', 'subscription_tier']:
                set_clauses.append(f"{key} = ${param_index}")
                params.append(value)
                param_index += 1
        
        if set_clauses:
            set_clauses.append("updated_at = datetime('now')")
            query = f"UPDATE organizations SET {', '.join(set_clauses)} WHERE id = $1"
            await self.db.execute(query, *params)
        
        return await self.get_organization(org_id)

    async def add_member(self, organization_id: int, user_id: int, role: str, invited_by: int):
        """Add a member to organization"""
        query = """
            INSERT OR REPLACE INTO organization_members (
                organization_id, user_id, role, status, invited_by, joined_at, updated_at
            ) VALUES ($1, $2, $3, 'active', $4, datetime('now'), datetime('now'))
        """
        
        await self.db.execute(query, organization_id, user_id, role, invited_by)

    async def get_member(self, organization_id: int, user_id: int):
        """Get organization member details"""
        query = """
            SELECT om.*, u.username, u.email 
            FROM organization_members om
            JOIN users u ON om.user_id = u.id
            WHERE om.organization_id = $1 AND om.user_id = $2 AND om.status = 'active'
        """
        return await self.db.fetch_one(query, organization_id, user_id)

    async def get_members(self, organization_id: int):
        """Get all organization members"""
        query = """
            SELECT om.user_id, u.username, u.email, u.display_name, om.role, om.status, 
                   om.joined_at, u.updated_at as last_active,
                   GROUP_CONCAT(t.name) as teams
            FROM organization_members om
            JOIN users u ON om.user_id = u.id
            LEFT JOIN team_members tm ON om.user_id = tm.user_id
            LEFT JOIN teams t ON tm.team_id = t.id AND t.organization_id = $1
            WHERE om.organization_id = $2 AND om.status = 'active'
            GROUP BY om.user_id
            ORDER BY om.role, u.username
        """
        return await self.db.fetch_all(query, organization_id, organization_id)

    async def get_member_count(self, organization_id: int) -> int:
        """Get organization member count"""
        query = """
            SELECT COUNT(*) as count FROM organization_members 
            WHERE organization_id = $1 AND status = 'active'
        """
        result = await self.db.fetch_one(query, organization_id)
        return result['count'] if result else 0

    async def get_owner_count(self, organization_id: int) -> int:
        """Get organization owner count"""
        query = """
            SELECT COUNT(*) as count FROM organization_members 
            WHERE organization_id = $1 AND role = 'owner' AND status = 'active'
        """
        result = await self.db.fetch_one(query, organization_id)
        return result['count'] if result else 0

    async def update_member_role(self, organization_id: int, user_id: int, role: str):
        """Update member role"""
        query = """
            UPDATE organization_members 
            SET role = $3, updated_at = datetime('now')
            WHERE organization_id = $1 AND user_id = $2
        """
        await self.db.execute(query, organization_id, user_id, role)
        return await self.get_member(organization_id, user_id)

    async def remove_member(self, organization_id: int, user_id: int):
        """Remove member from organization"""
        # Remove from teams first
        await self.db.execute("""
            DELETE FROM team_members 
            WHERE user_id = $1 AND team_id IN (
                SELECT id FROM teams WHERE organization_id = $2
            )
        """, user_id, organization_id)
        
        # Remove from organization
        await self.db.execute("""
            DELETE FROM organization_members 
            WHERE organization_id = $1 AND user_id = $2
        """, organization_id, user_id)

    async def create_invitation(
        self, 
        organization_id: int, 
        email: str, 
        role: str,
        team_id: Optional[int] = None,
        message: Optional[str] = None,
        invited_by: int = None
    ):
        """Create invitation for new member"""
        invitation_uuid = str(uuid.uuid4())
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(days=7)  # 7 days expiry
        
        query = """
            INSERT INTO invitations (
                uuid, organization_id, email, role, team_id, message, token,
                status, invited_by, expires_at, created_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, 'pending', $8, $9, datetime('now')
            )
        """
        
        await self.db.execute(query, 
            invitation_uuid, organization_id, email, role, team_id, 
            message, token, invited_by, expires_at.isoformat()
        )
        
        # Return invitation details
        invitation = await self.db.fetch_one(
            "SELECT * FROM invitations WHERE uuid = $1", invitation_uuid
        )
        return invitation

    async def create_team(
        self,
        organization_id: int,
        name: str,
        description: Optional[str] = None,
        color: str = "#3B82F6",
        is_default: bool = False,
        created_by: int = None
    ):
        """Create a new team"""
        team_uuid = str(uuid.uuid4())
        
        query = """
            INSERT INTO teams (
                uuid, organization_id, name, description, color, is_default,
                created_by, created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, datetime('now'), datetime('now')
            )
        """
        
        await self.db.execute(query, 
            team_uuid, organization_id, name, description, color, 
            1 if is_default else 0, created_by
        )
        
        team = await self.db.fetch_one("SELECT * FROM teams WHERE uuid = $1", team_uuid)
        return team

    async def get_teams(self, organization_id: int, user_id: int = 0):
        """Get organization teams"""
        query = """
            SELECT t.*, 
                   COUNT(tm.user_id) as member_count,
                   CASE WHEN tm_user.user_id IS NOT NULL THEN 1 ELSE 0 END as is_member,
                   tm_user.role as user_role
            FROM teams t
            LEFT JOIN team_members tm ON t.id = tm.team_id
            LEFT JOIN team_members tm_user ON t.id = tm_user.team_id AND tm_user.user_id = $2
            WHERE t.organization_id = $1
            GROUP BY t.id
            ORDER BY t.is_default DESC, t.name
        """
        return await self.db.fetch_all(query, organization_id, user_id)

    async def add_team_member(self, team_id: int, user_id: int, role: str = "member", added_by: int = None):
        """Add member to team"""
        query = """
            INSERT OR REPLACE INTO team_members (
                team_id, user_id, role, added_by, joined_at
            ) VALUES ($1, $2, $3, $4, datetime('now'))
        """
        
        await self.db.execute(query, team_id, user_id, role, added_by)

    async def get_team_member(self, team_id: int, user_id: int):
        """Get team member details"""
        query = """
            SELECT tm.*, u.username, u.email 
            FROM team_members tm
            JOIN users u ON tm.user_id = u.id
            WHERE tm.team_id = $1 AND tm.user_id = $2
        """
        return await self.db.fetch_one(query, team_id, user_id)

    async def remove_team_member(self, team_id: int, user_id: int):
        """Remove member from team"""
        query = "DELETE FROM team_members WHERE team_id = $1 AND user_id = $2"
        await self.db.execute(query, team_id, user_id)

    async def get_recent_activity(self, organization_id: int, limit: int = 10):
        """Get recent activity for organization"""
        query = """
            SELECT al.action, al.resource_type, al.resource_id, al.details, 
                   al.created_at, u.username, u.email
            FROM audit_logs al
            JOIN users u ON al.user_id = u.id
            WHERE al.organization_id = $1
            ORDER BY al.created_at DESC
            LIMIT $2
        """
        return await self.db.fetch_all(query, organization_id, limit)