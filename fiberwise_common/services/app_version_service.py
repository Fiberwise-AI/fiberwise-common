"""
App Version Service - manages app versions, deployments, and rollbacks
"""

import json
import logging
import uuid
import yaml
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from .base_service import BaseService, ServiceError, NotFoundError, ValidationError
from ..database.query_adapter import QueryAdapter, ParameterStyle

logger = logging.getLogger(__name__)


class AppVersionService(BaseService):
    """
    Service for managing app versions, deployments, and rollbacks.
    """
    
    def __init__(self, db_provider):
        """
        Initialize with a database provider.
        
        Args:
            db_provider: Database provider (SQLiteProvider, PostgreSQLProvider, etc.)
        """
        super().__init__(db_provider)
        self.query_adapter = QueryAdapter(ParameterStyle.SQLITE)

    async def get_app_versions(
        self,
        app_id: str,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get all versions for a specific app.
        
        Args:
            app_id: App ID
            status: Filter by status (draft, active, deprecated, etc.)
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            List of app version records
        """
        query_parts = ["""
            SELECT 
                app_version_id,
                app_id,
                version,
                manifest_yaml,
                status,
                changelog,
                is_active,
                created_by,
                created_at,
                updated_at,
                deployed_at
            FROM app_versions
            WHERE app_id = ?
        """]
        params = [app_id]
        
        if status:
            query_parts.append("AND status = ?")
            params.append(status)
        
        query_parts.append("ORDER BY created_at DESC")
        query_parts.append(f"LIMIT {limit} OFFSET {offset}")
        
        query = " ".join(query_parts)
        versions = await self.db.fetchall(query, *params)
        
        # Process results
        result = []
        for version in versions:
            version_dict = dict(version)
            result.append(version_dict)
        
        return result

    async def get_version_by_id(self, version_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific app version by ID.
        
        Args:
            version_id: The version ID to retrieve
            
        Returns:
            Version record or None if not found
        """
        query = "SELECT * FROM app_versions WHERE app_version_id = ?"
        version = await self.db.fetchone(query, version_id)
        
        if not version:
            return None
        
        return dict(version)

    async def get_active_version(self, app_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the currently active version for an app.
        
        Args:
            app_id: App ID
            
        Returns:
            Active version record or None if no active version
        """
        query = """
            SELECT * FROM app_versions 
            WHERE app_id = ? AND is_active = 1
            ORDER BY created_at DESC
            LIMIT 1
        """
        version = await self.db.fetchone(query, app_id)
        
        if not version:
            return None
        
        return dict(version)

    async def create_version(
        self,
        app_id: str,
        version_data: Dict[str, Any],
        created_by: int
    ) -> Dict[str, Any]:
        """
        Create a new app version.
        
        Args:
            app_id: App ID
            version_data: Version data
            created_by: User ID creating the version
            
        Returns:
            Created version record
        """
        version_id = str(uuid.uuid4())
        
        # Validate manifest YAML
        try:
            if version_data.get('manifest_yaml'):
                yaml.safe_load(version_data['manifest_yaml'])
        except yaml.YAMLError as e:
            raise ValidationError(f"Invalid manifest YAML: {str(e)}")
        
        query = """
            INSERT INTO app_versions (
                app_version_id, app_id, version, manifest_yaml, status,
                changelog, is_active, created_by, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        now = datetime.now().isoformat()
        
        await self.db.execute(query, (
            version_id,
            app_id,
            version_data.get('version'),
            version_data.get('manifest_yaml', ''),
            version_data.get('status', 'draft'),
            version_data.get('changelog', ''),
            0,  # is_active defaults to False
            created_by,
            now,
            now
        ))
        
        return await self.get_version_by_id(version_id)

    async def update_version(
        self,
        version_id: str,
        update_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Update an app version.
        
        Args:
            version_id: Version ID to update
            update_data: Updated version data
            
        Returns:
            Updated version record or None if not found
        """
        # Check if version exists
        existing = await self.get_version_by_id(version_id)
        if not existing:
            return None
        
        # Validate manifest YAML if provided
        if 'manifest_yaml' in update_data:
            try:
                yaml.safe_load(update_data['manifest_yaml'])
            except yaml.YAMLError as e:
                raise ValidationError(f"Invalid manifest YAML: {str(e)}")
        
        # Build dynamic update query
        update_fields = []
        params = []
        
        for field in ['status', 'changelog', 'manifest_yaml', 'is_active']:
            if field in update_data:
                update_fields.append(f"{field} = ?")
                params.append(update_data[field])
        
        if not update_fields:
            return existing
        
        # Add updated_at
        update_fields.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(version_id)
        
        query = f"""
            UPDATE app_versions 
            SET {', '.join(update_fields)}
            WHERE app_version_id = ?
        """
        
        await self.db.execute(query, *params)
        
        return await self.get_version_by_id(version_id)

    async def deploy_version(
        self,
        version_id: str,
        deployment_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Deploy a version (make it active).
        
        Args:
            version_id: Version ID to deploy
            deployment_data: Deployment configuration
            
        Returns:
            Deployment result
        """
        version = await self.get_version_by_id(version_id)
        if not version:
            raise NotFoundError(f"Version {version_id} not found")
        
        app_id = version['app_id']
        
        # Start transaction
        try:
            # Deactivate current active version
            await self.db.execute(
                "UPDATE app_versions SET is_active = 0 WHERE app_id = ? AND is_active = 1",
                app_id
            )
            
            # Activate new version
            now = datetime.now().isoformat()
            await self.db.execute(
                "UPDATE app_versions SET is_active = 1, deployed_at = ?, status = 'active' WHERE app_version_id = ?",
                now, version_id
            )
            
            return {
                'success': True,
                'version_id': version_id,
                'deployed_at': now,
                'message': f"Version {version['version']} deployed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error deploying version {version_id}: {e}")
            raise ServiceError(f"Failed to deploy version: {str(e)}")

    async def rollback_to_version(
        self,
        current_version_id: str,
        target_version_id: str,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Rollback from current version to a target version.
        
        Args:
            current_version_id: Current active version ID
            target_version_id: Target version ID to rollback to
            reason: Reason for rollback
            
        Returns:
            Rollback result
        """
        current_version = await self.get_version_by_id(current_version_id)
        target_version = await self.get_version_by_id(target_version_id)
        
        if not current_version:
            raise NotFoundError(f"Current version {current_version_id} not found")
        if not target_version:
            raise NotFoundError(f"Target version {target_version_id} not found")
        
        if current_version['app_id'] != target_version['app_id']:
            raise ValidationError("Versions must belong to the same app")
        
        app_id = current_version['app_id']
        
        try:
            # Deactivate current version
            await self.db.execute(
                "UPDATE app_versions SET is_active = 0, status = 'rolled_back' WHERE app_version_id = ?",
                current_version_id
            )
            
            # Activate target version
            now = datetime.now().isoformat()
            await self.db.execute(
                "UPDATE app_versions SET is_active = 1, deployed_at = ?, status = 'active' WHERE app_version_id = ?",
                now, target_version_id
            )
            
            return {
                'success': True,
                'old_version_id': current_version_id,
                'new_version_id': target_version_id,
                'rollback_timestamp': now,
                'message': f"Successfully rolled back from {current_version['version']} to {target_version['version']}",
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Error rolling back from {current_version_id} to {target_version_id}: {e}")
            raise ServiceError(f"Failed to rollback: {str(e)}")

    async def compare_versions(
        self,
        version_a_id: str,
        version_b_id: str
    ) -> Dict[str, Any]:
        """
        Compare two app versions.
        
        Args:
            version_a_id: First version ID
            version_b_id: Second version ID
            
        Returns:
            Comparison result with differences
        """
        version_a = await self.get_version_by_id(version_a_id)
        version_b = await self.get_version_by_id(version_b_id)
        
        if not version_a:
            raise NotFoundError(f"Version {version_a_id} not found")
        if not version_b:
            raise NotFoundError(f"Version {version_b_id} not found")
        
        # Parse manifests
        try:
            manifest_a = yaml.safe_load(version_a['manifest_yaml']) if version_a['manifest_yaml'] else {}
            manifest_b = yaml.safe_load(version_b['manifest_yaml']) if version_b['manifest_yaml'] else {}
        except yaml.YAMLError as e:
            raise ValidationError(f"Error parsing manifest YAML: {str(e)}")
        
        # Simple diff (you could use more sophisticated diffing libraries)
        diff = self._generate_manifest_diff(manifest_a, manifest_b)
        
        return {
            'version_a': version_a,
            'version_b': version_b,
            'manifest_diff': diff,
            'summary': {
                'has_changes': len(diff) > 0,
                'change_count': len(diff),
                'version_a_newer': version_a['created_at'] > version_b['created_at']
            }
        }

    def _generate_manifest_diff(self, manifest_a: Dict, manifest_b: Dict) -> List[Dict[str, Any]]:
        """
        Generate a simple diff between two manifest dictionaries.
        
        Args:
            manifest_a: First manifest
            manifest_b: Second manifest
            
        Returns:
            List of differences
        """
        differences = []
        
        # Get all keys from both manifests
        all_keys = set(manifest_a.keys()) | set(manifest_b.keys())
        
        for key in all_keys:
            if key not in manifest_a:
                differences.append({
                    'type': 'added',
                    'key': key,
                    'value': manifest_b[key]
                })
            elif key not in manifest_b:
                differences.append({
                    'type': 'removed',
                    'key': key,
                    'value': manifest_a[key]
                })
            elif manifest_a[key] != manifest_b[key]:
                differences.append({
                    'type': 'modified',
                    'key': key,
                    'old_value': manifest_a[key],
                    'new_value': manifest_b[key]
                })
        
        return differences

    async def delete_version(self, version_id: str) -> bool:
        """
        Delete an app version (soft delete - mark as deleted).
        
        Args:
            version_id: Version ID to delete
            
        Returns:
            True if deleted successfully
        """
        version = await self.get_version_by_id(version_id)
        if not version:
            return False
        
        if version['is_active']:
            raise ValidationError("Cannot delete active version. Deploy another version first.")
        
        # Soft delete by updating status
        await self.db.execute(
            "UPDATE app_versions SET status = 'deleted', updated_at = ? WHERE app_version_id = ?",
            datetime.now().isoformat(), version_id
        )
        
        return True

    async def get_version_metrics(self, version_id: str) -> Dict[str, Any]:
        """
        Get metrics for a specific version.
        
        Args:
            version_id: Version ID
            
        Returns:
            Version metrics
        """
        version = await self.get_version_by_id(version_id)
        if not version:
            raise NotFoundError(f"Version {version_id} not found")
        
        # Basic metrics from database
        metrics = {
            'version_id': version_id,
            'deployment_count': 1 if version['deployed_at'] else 0,
            'rollback_count': 0,  # Would need additional tracking
            'active_duration_hours': None,
            'error_rate': None,
            'performance_metrics': None,
            'last_deployed': version['deployed_at'],
            'last_rolled_back': None
        }
        
        # Calculate active duration if version was deployed
        if version['deployed_at'] and version['is_active']:
            deployed_at = datetime.fromisoformat(version['deployed_at'])
            duration = datetime.now() - deployed_at
            metrics['active_duration_hours'] = duration.total_seconds() / 3600
        
        return metrics
