"""
Unified Executions Service - Query both agent activations and function executions
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from fiberwise_common import DatabaseProvider

logger = logging.getLogger(__name__)


class UnifiedExecutionsService:
    """Service for querying executions across both agents and functions"""
    
    def __init__(self, db_provider: DatabaseProvider):
        self.db = db_provider

    async def get_all_executions(
        self, 
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
        execution_type: Optional[str] = None,  # 'agent' or 'function'
        entity_id: Optional[str] = None,
        created_by: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get unified view of all executions (agents and functions).
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            status: Filter by status ('queued', 'running', 'completed', 'failed')
            execution_type: Filter by type ('agent' or 'function')
            entity_id: Filter by specific agent_id or function_id
            created_by: Filter by user who created the execution
            start_date: Filter executions after this date (ISO format)
            end_date: Filter executions before this date (ISO format)
            
        Returns:
            List of unified execution records
        """
        try:
            # Build query with filters
            query_parts = ["SELECT * FROM unified_executions WHERE 1=1"]
            params = []
            param_index = 1
            
            if status:
                query_parts.append(f"AND status = ${param_index}")
                params.append(status)
                param_index += 1
                
            if execution_type:
                query_parts.append(f"AND execution_type = ${param_index}")
                params.append(execution_type)
                param_index += 1
                
            if entity_id:
                query_parts.append(f"AND entity_id = ${param_index}")
                params.append(entity_id)
                param_index += 1
                
            if created_by:
                query_parts.append(f"AND created_by = ${param_index}")
                params.append(created_by)
                param_index += 1
                
            if start_date:
                query_parts.append(f"AND started_at >= ${param_index}")
                params.append(start_date)
                param_index += 1
                
            if end_date:
                query_parts.append(f"AND started_at <= ${param_index}")
                params.append(end_date)
                param_index += 1
            
            # Add ordering and pagination
            query_parts.append("ORDER BY started_at DESC")
            query_parts.append(f"LIMIT ${param_index} OFFSET ${param_index + 1}")
            params.extend([limit, offset])
            
            # Execute query
            query = " ".join(query_parts)
            results = await self.db.fetch_all(query, *params)
            
            # Process results
            executions = []
            for row in results:
                execution = dict(row)
                
                # Parse JSON fields if they're strings
                for field in ['input_data', 'output_data', 'metadata', 'context', 'notes']:
                    if execution.get(field) and isinstance(execution[field], str):
                        try:
                            import json
                            execution[field] = json.loads(execution[field])
                        except:
                            pass  # Keep as string if not valid JSON
                
                executions.append(execution)
            
            return executions
            
        except Exception as e:
            logger.error(f"Error fetching unified executions: {str(e)}")
            return []

    async def get_execution_stats(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics across all execution types.
        
        Returns:
            Dictionary with execution statistics
        """
        try:
            # Base query for stats
            where_clause = ""
            params = []
            
            if start_date or end_date:
                conditions = []
                param_index = 1
                
                if start_date:
                    conditions.append(f"started_at >= ${param_index}")
                    params.append(start_date)
                    param_index += 1
                    
                if end_date:
                    conditions.append(f"started_at <= ${param_index}")
                    params.append(end_date)
                    param_index += 1
                
                where_clause = "WHERE " + " AND ".join(conditions)
            
            # Get overall stats
            query = f"""
                SELECT 
                    execution_type,
                    status,
                    COUNT(*) as count,
                    AVG(duration_ms) as avg_duration_ms,
                    MIN(started_at) as first_execution,
                    MAX(started_at) as last_execution
                FROM unified_executions
                {where_clause}
                GROUP BY execution_type, status
                ORDER BY execution_type, status
            """
            
            results = await self.db.fetch_all(query, *params)
            
            # Process stats
            stats = {
                'total': 0,
                'by_type': {},
                'by_status': {},
                'success_rate': 0,
                'avg_duration_ms': 0
            }
            
            total_count = 0
            completed_count = 0
            total_duration = 0
            duration_count = 0
            
            for row in results:
                row_dict = dict(row)
                execution_type = row_dict['execution_type']
                status = row_dict['status']
                count = row_dict['count']
                avg_duration = row_dict['avg_duration_ms'] or 0
                
                # Initialize type stats if needed
                if execution_type not in stats['by_type']:
                    stats['by_type'][execution_type] = {
                        'total': 0,
                        'completed': 0,
                        'failed': 0,
                        'running': 0,
                        'queued': 0
                    }
                
                # Update type stats
                stats['by_type'][execution_type]['total'] += count
                stats['by_type'][execution_type][status] = count
                
                # Update status stats
                if status not in stats['by_status']:
                    stats['by_status'][status] = 0
                stats['by_status'][status] += count
                
                # Track totals
                total_count += count
                if status == 'completed':
                    completed_count += count
                
                # Track duration
                if avg_duration > 0:
                    total_duration += avg_duration * count
                    duration_count += count
            
            # Calculate overall stats
            stats['total'] = total_count
            stats['success_rate'] = (completed_count / total_count * 100) if total_count > 0 else 0
            stats['avg_duration_ms'] = (total_duration / duration_count) if duration_count > 0 else 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting execution stats: {str(e)}")
            return {
                'total': 0,
                'by_type': {},
                'by_status': {},
                'success_rate': 0,
                'avg_duration_ms': 0
            }

    async def get_recent_executions_by_user(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent executions for a specific user."""
        return await self.get_all_executions(
            limit=limit,
            created_by=user_id
        )

    async def get_failed_executions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent failed executions across all types."""
        return await self.get_all_executions(
            limit=limit,
            status='failed'
        )

    async def get_execution_by_id(self, execution_id: str, execution_type: str = None) -> Optional[Dict[str, Any]]:
        """
        Get a specific execution by ID.
        
        Args:
            execution_id: The execution ID to find
            execution_type: Optional hint about type ('agent' or 'function')
            
        Returns:
            Execution record or None if not found
        """
        try:
            # If type is specified, query the specific table
            if execution_type == 'agent':
                query = "SELECT *, 'agent' as execution_type FROM agent_activations WHERE activation_id = $1"
            elif execution_type == 'function':
                query = "SELECT *, 'function' as execution_type FROM function_executions WHERE execution_id = $1"
            else:
                # Search both tables
                query = "SELECT * FROM unified_executions WHERE execution_id = $1"
            
            result = await self.db.fetch_one(query, execution_id)
            
            if result:
                execution = dict(result)
                
                # Parse JSON fields
                for field in ['input_data', 'output_data', 'metadata', 'context', 'notes']:
                    if execution.get(field) and isinstance(execution[field], str):
                        try:
                            import json
                            execution[field] = json.loads(execution[field])
                        except:
                            pass
                
                return execution
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting execution {execution_id}: {str(e)}")
            return None
