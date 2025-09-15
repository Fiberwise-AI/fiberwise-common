"""
App Migration Service

Handles validation and migration of app manifest changes to ensure data integrity
and prevent dangerous changes like user isolation modifications.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import yaml
from enum import Enum

from fiberwise_common import DatabaseProvider

logger = logging.getLogger(__name__)


class MigrationRisk(Enum):
    """Risk levels for different types of migrations"""
    SAFE = "safe"           # No data impact, can be auto-applied
    LOW = "low"             # Minor changes, reversible  
    MEDIUM = "medium"       # Potentially breaking, needs review
    HIGH = "high"           # Dangerous changes, needs manual review
    BLOCKED = "blocked"     # Not allowed, would cause data loss/corruption


class MigrationIssue:
    """Represents a migration issue or warning"""
    
    def __init__(self, risk: MigrationRisk, field_path: str, message: str, 
                 current_value: Any = None, new_value: Any = None):
        self.risk = risk
        self.field_path = field_path
        self.message = message
        self.current_value = current_value
        self.new_value = new_value
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk": self.risk.value,
            "field_path": self.field_path,
            "message": self.message,
            "current_value": self.current_value,
            "new_value": self.new_value
        }


class AppMigrationService:
    """Service for validating and managing app manifest migrations"""
    
    def __init__(self, db: DatabaseProvider):
        self.db = db
    
    async def validate_app_update(self, app_id: str, new_manifest: Dict[str, Any]) -> Tuple[bool, List[MigrationIssue]]:
        """
        Validate if an app manifest update is safe to apply.
        
        Returns:
            (is_safe: bool, issues: List[MigrationIssue])
        """
        issues = []
        
        # Get current manifest
        current_manifest = await self._get_current_manifest(app_id)
        if not current_manifest:
            # New app - always safe
            return True, []
        
        # Check user isolation changes
        isolation_issues = await self._check_user_isolation_changes(
            app_id, current_manifest, new_manifest
        )
        issues.extend(isolation_issues)
        
        # Check model changes
        model_issues = await self._check_model_changes(
            current_manifest, new_manifest
        )
        issues.extend(model_issues)
        
        # Check field changes
        field_issues = await self._check_field_changes(
            current_manifest, new_manifest
        )
        issues.extend(field_issues)
        
        # Determine if the migration is safe
        has_blocked = any(issue.risk == MigrationRisk.BLOCKED for issue in issues)
        has_high_risk = any(issue.risk == MigrationRisk.HIGH for issue in issues)
        
        is_safe = not (has_blocked or has_high_risk)
        
        return is_safe, issues
    
    async def _get_current_manifest(self, app_id: str) -> Optional[Dict[str, Any]]:
        """Get the current manifest for an app"""
        query = """
            SELECT manifest_yaml FROM app_versions 
            WHERE app_id = $1 
            ORDER BY created_at DESC 
            LIMIT 1
        """
        result = await self.db.fetch_one(query, app_id)
        
        if not result or not result["manifest_yaml"]:
            return None
        
        try:
            return yaml.safe_load(result["manifest_yaml"])
        except Exception as e:
            logger.error(f"Failed to parse current manifest for app {app_id}: {e}")
            return None
    
    async def _check_user_isolation_changes(
        self, app_id: str, current: Dict[str, Any], new: Dict[str, Any]
    ) -> List[MigrationIssue]:
        """Check for dangerous user isolation changes"""
        issues = []
        
        current_app = current.get("app", {})
        new_app = new.get("app", {})
        
        current_isolation = current_app.get("user_isolation", "enforced")
        new_isolation = new_app.get("user_isolation", "enforced")
        
        if current_isolation != new_isolation:
            # Check if there's existing data
            data_count = await self._count_app_data(app_id)
            
            if data_count > 0:
                # Changing isolation on app with existing data is risky
                if current_isolation == "disabled" and new_isolation == "enforced":
                    # Going from shared to isolated - BLOCKED (would hide existing data)
                    issues.append(MigrationIssue(
                        risk=MigrationRisk.BLOCKED,
                        field_path="app.user_isolation",
                        message=f"Cannot change from 'disabled' to 'enforced' on app with {data_count} existing items. This would hide shared data and break user access.",
                        current_value=current_isolation,
                        new_value=new_isolation
                    ))
                elif current_isolation == "enforced" and new_isolation == "disabled":
                    # Going from isolated to shared - HIGH RISK (exposes private data)
                    issues.append(MigrationIssue(
                        risk=MigrationRisk.HIGH,
                        field_path="app.user_isolation",
                        message=f"Changing from 'enforced' to 'disabled' will make all {data_count} user items visible to everyone. This may expose private data.",
                        current_value=current_isolation,
                        new_value=new_isolation
                    ))
                else:
                    # Other isolation changes
                    issues.append(MigrationIssue(
                        risk=MigrationRisk.MEDIUM,
                        field_path="app.user_isolation",
                        message=f"User isolation change from '{current_isolation}' to '{new_isolation}' may affect data visibility for {data_count} items.",
                        current_value=current_isolation,
                        new_value=new_isolation
                    ))
            else:
                # No existing data - safe to change
                issues.append(MigrationIssue(
                    risk=MigrationRisk.SAFE,
                    field_path="app.user_isolation",
                    message=f"User isolation change from '{current_isolation}' to '{new_isolation}' (no existing data).",
                    current_value=current_isolation,
                    new_value=new_isolation
                ))
        
        return issues
    
    async def _count_app_data(self, app_id: str) -> int:
        """Count total data items for an app"""
        query = "SELECT COUNT(*) FROM app_model_items WHERE app_id = $1"
        return await self.db.fetch_val(query, app_id) or 0
    
    async def _check_model_changes(
        self, current: Dict[str, Any], new: Dict[str, Any]
    ) -> List[MigrationIssue]:
        """Check for model-level changes"""
        issues = []
        
        current_models = {m.get("model_slug"): m for m in current.get("app", {}).get("models", [])}
        new_models = {m.get("model_slug"): m for m in new.get("app", {}).get("models", [])}
        
        # Check for removed models
        for model_slug in current_models:
            if model_slug not in new_models:
                issues.append(MigrationIssue(
                    risk=MigrationRisk.HIGH,
                    field_path=f"app.models[{model_slug}]",
                    message=f"Model '{model_slug}' removed - existing data would be orphaned",
                    current_value=current_models[model_slug],
                    new_value=None
                ))
        
        # Check for new models (usually safe)
        for model_slug in new_models:
            if model_slug not in current_models:
                issues.append(MigrationIssue(
                    risk=MigrationRisk.SAFE,
                    field_path=f"app.models[{model_slug}]",
                    message=f"New model '{model_slug}' added",
                    current_value=None,
                    new_value=new_models[model_slug]
                ))
        
        return issues
    
    async def _check_field_changes(
        self, current: Dict[str, Any], new: Dict[str, Any]
    ) -> List[MigrationIssue]:
        """Check for field-level changes"""
        issues = []
        
        current_models = {m.get("model_slug"): m for m in current.get("app", {}).get("models", [])}
        new_models = {m.get("model_slug"): m for m in new.get("app", {}).get("models", [])}
        
        # Check field changes for existing models
        for model_slug in current_models:
            if model_slug not in new_models:
                continue  # Already handled in model changes
            
            current_fields = {f.get("field_column"): f for f in current_models[model_slug].get("fields", [])}
            new_fields = {f.get("field_column"): f for f in new_models[model_slug].get("fields", [])}
            
            # Check for removed fields
            for field_name in current_fields:
                if field_name not in new_fields:
                    issues.append(MigrationIssue(
                        risk=MigrationRisk.MEDIUM,
                        field_path=f"app.models[{model_slug}].fields[{field_name}]",
                        message=f"Field '{field_name}' removed from model '{model_slug}' - data would be lost",
                        current_value=current_fields[field_name],
                        new_value=None
                    ))
            
            # Check for field type changes
            for field_name in new_fields:
                if field_name in current_fields:
                    current_type = current_fields[field_name].get("type")
                    new_type = new_fields[field_name].get("type")
                    
                    if current_type != new_type:
                        # Type changes can be risky
                        risk = MigrationRisk.MEDIUM
                        if current_type in ["uuid", "string"] and new_type in ["integer", "float"]:
                            risk = MigrationRisk.HIGH  # Likely to cause conversion errors
                        
                        issues.append(MigrationIssue(
                            risk=risk,
                            field_path=f"app.models[{model_slug}].fields[{field_name}].type",
                            message=f"Field type change from '{current_type}' to '{new_type}' may cause data conversion issues",
                            current_value=current_type,
                            new_value=new_type
                        ))
        
        return issues


# Utility function for easy access
async def validate_app_migration(db: DatabaseProvider, app_id: str, new_manifest: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Utility function to validate an app migration.
    
    Returns:
        (is_safe: bool, issues: List[Dict])
    """
    service = AppMigrationService(db)
    is_safe, issues = await service.validate_app_update(app_id, new_manifest)
    return is_safe, [issue.to_dict() for issue in issues]
