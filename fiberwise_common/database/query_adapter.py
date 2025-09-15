"""
SQL Query Adapter for cross-database compatibility.
Converts parameter placeholders between different database systems.
"""

import re
from typing import Tuple, Any, List
from enum import Enum


class ParameterStyle(Enum):
    """Database parameter placeholder styles."""
    POSTGRESQL = "postgresql"  # $1, $2, $3
    MYSQL = "mysql"           # %s, %s, %s
    SQLITE = "sqlite"         # ?, ?, ?
    MSSQL = "mssql"          # ?, ?, ? (same as SQLite)
    NAMED = "named"          # :param1, :param2


class QueryAdapter:
    """Converts SQL queries and parameters between different database systems."""
    
    def __init__(self, target_style: ParameterStyle):
        self.target_style = target_style
    
    def convert_query(self, query: str, source_style: ParameterStyle = ParameterStyle.POSTGRESQL) -> str:
        """
        Convert query from source parameter style to target parameter style.
        Also converts SQL functions for cross-database compatibility.
        
        Args:
            query: SQL query with source-style parameters
            source_style: Source parameter style (default: PostgreSQL)
            
        Returns:
            Converted query string
        """
        # First handle parameter style conversion
        converted_query = query
        if source_style != self.target_style:
            if source_style == ParameterStyle.POSTGRESQL:
                converted_query = self._convert_from_postgresql(converted_query)
            elif source_style == ParameterStyle.MYSQL:
                converted_query = self._convert_from_mysql(converted_query)
            elif source_style == ParameterStyle.SQLITE:
                converted_query = self._convert_from_sqlite(converted_query)
            elif source_style == ParameterStyle.NAMED:
                converted_query = self._convert_from_named(converted_query)
            else:
                raise ValueError(f"Unsupported source parameter style: {source_style}")
        
        # Then handle SQL function conversions
        converted_query = self._convert_sql_functions(converted_query, source_style)
        
        return converted_query
    
    def _convert_from_postgresql(self, query: str) -> str:
        """Convert from PostgreSQL $1, $2, $3 style."""
        converted = query
        
        # Handle PostgreSQL array syntax first (before parameter conversion)
        if self.target_style == ParameterStyle.SQLITE:
            converted = self._convert_postgresql_arrays_to_sqlite(converted)
        
        # Then handle regular parameter conversion
        if self.target_style == ParameterStyle.SQLITE or self.target_style == ParameterStyle.MSSQL:
            # Replace $1, $2, etc. with ?
            converted = re.sub(r'\$\d+', '?', converted)
        elif self.target_style == ParameterStyle.MYSQL:
            # Replace $1, $2, etc. with %s
            converted = re.sub(r'\$\d+', '%s', converted)
        elif self.target_style == ParameterStyle.NAMED:
            # Replace $1, $2, etc. with :param1, :param2
            def replace_param(match):
                param_num = match.group(0)[1:]  # Remove $
                return f':param{param_num}'
            converted = re.sub(r'\$\d+', replace_param, converted)
        
        return converted
    
    def _convert_postgresql_arrays_to_sqlite(self, query: str) -> str:
        """
        Convert PostgreSQL array syntax to SQLite-compatible syntax.
        
        Converts patterns like:
        - column = ANY($1::uuid[]) -> column IN (SELECT value FROM json_each(?))
        - column = ANY($1::text[]) -> column IN (SELECT value FROM json_each(?))
        
        Note: This requires parameters to be passed as JSON arrays to SQLite
        """
        # Pattern to match: column = ANY($n::type[])
        array_pattern = r'(\w+)\s*=\s*ANY\(\$(\d+)::(\w+)\[\]\)'
        
        def replace_any_array(match):
            column_name = match.group(1)
            param_num = match.group(2)  # We'll replace this with ? later
            data_type = match.group(3)  # uuid, text, etc.
            
            # Convert to SQLite's JSON array expansion
            # The parameter will be converted to ? by the regular parameter conversion
            return f"{column_name} IN (SELECT value FROM json_each($${param_num}))"
        
        converted = re.sub(array_pattern, replace_any_array, query)
        return converted
    
    def _convert_array_params_for_sqlite(self, params: Any) -> Any:
        """
        Convert array parameters to JSON format for SQLite compatibility.
        
        When using ANY($1::type[]) syntax, PostgreSQL expects an array parameter.
        For SQLite, we convert these arrays to JSON strings that can be used
        with json_each() function.
        """
        import json
        
        if not isinstance(params, (list, tuple)):
            return params
            
        converted_params = []
        for param in params:
            if isinstance(param, (list, tuple)):
                # Convert array to JSON string for SQLite
                json_param = json.dumps([str(item) for item in param])
                converted_params.append(json_param)
            else:
                converted_params.append(param)
        
        return tuple(converted_params) if isinstance(params, tuple) else converted_params
    
    def _convert_from_mysql(self, query: str) -> str:
        """Convert from MySQL %s style."""
        if self.target_style == ParameterStyle.SQLITE or self.target_style == ParameterStyle.MSSQL:
            # Replace %s with ?
            return query.replace('%s', '?')
        elif self.target_style == ParameterStyle.POSTGRESQL:
            # Replace %s with $1, $2, etc.
            counter = 1
            def replace_param(match):
                nonlocal counter
                result = f'${counter}'
                counter += 1
                return result
            return re.sub(r'%s', replace_param, query)
        return query
    
    def _convert_from_sqlite(self, query: str) -> str:
        """Convert from SQLite ? style."""
        if self.target_style == ParameterStyle.POSTGRESQL:
            # Replace ? with $1, $2, etc.
            counter = 1
            def replace_param(match):
                nonlocal counter
                result = f'${counter}'
                counter += 1
                return result
            return re.sub(r'\?', replace_param, query)
        elif self.target_style == ParameterStyle.MYSQL:
            # Replace ? with %s
            return query.replace('?', '%s')
        return query
    
    def _convert_from_named(self, query: str) -> str:
        """Convert from named :param style."""
        if self.target_style == ParameterStyle.SQLITE or self.target_style == ParameterStyle.MSSQL:
            # Replace :param with ?
            return re.sub(r':\w+', '?', query)
        elif self.target_style == ParameterStyle.MYSQL:
            # Replace :param with %s
            return re.sub(r':\w+', '%s', query)
        elif self.target_style == ParameterStyle.POSTGRESQL:
            # Replace :param with $1, $2, etc.
            counter = 1
            def replace_param(match):
                nonlocal counter
                result = f'${counter}'
                counter += 1
                return result
            return re.sub(r':\w+', replace_param, query)
        return query
    
    def _convert_sql_functions(self, query: str, source_style: ParameterStyle) -> str:
        """
        Convert SQL functions for cross-database compatibility.
        
        Args:
            query: SQL query string
            source_style: Source database style
            
        Returns:
            Query with converted SQL functions
        """
        converted = query
        
        # Convert PostgreSQL functions to target database equivalents
        if source_style == ParameterStyle.POSTGRESQL and self.target_style == ParameterStyle.SQLITE:
            # PostgreSQL -> SQLite conversions
            converted = re.sub(r'\bNOW\(\)', 'CURRENT_TIMESTAMP', converted)
            converted = re.sub(r'\bILIKE\b', 'LIKE', converted)  # SQLite LIKE is case-insensitive by default
            # Add more PostgreSQL -> SQLite conversions as needed
            # converted = re.sub(r'\bBOOLEAN\b', 'INTEGER', converted)
            # converted = re.sub(r'\bSERIAL\b', 'INTEGER PRIMARY KEY AUTOINCREMENT', converted)
        
        elif source_style == ParameterStyle.POSTGRESQL and self.target_style == ParameterStyle.MYSQL:
            # PostgreSQL -> MySQL conversions
            converted = re.sub(r'\bNOW\(\)', 'NOW()', converted)  # Same function name
            converted = re.sub(r'\bILIKE\b', 'LIKE', converted)   # MySQL LIKE is case-insensitive by default
            # Add more PostgreSQL -> MySQL conversions as needed
        
        # Add more source/target combinations as needed
        # elif source_style == ParameterStyle.MYSQL and self.target_style == ParameterStyle.SQLITE:
        #     # MySQL -> SQLite conversions
        #     pass
        
        return converted
    
    def convert_parameters(self, params: Any, source_style: ParameterStyle = ParameterStyle.POSTGRESQL) -> Any:
        """
        Convert parameters if needed for the target database.
        
        Args:
            params: Parameter values (tuple, list, or dict)
            source_style: Source parameter style
            
        Returns:
            Converted parameters
        """
        # Handle PostgreSQL array parameters for SQLite
        if (source_style == ParameterStyle.POSTGRESQL and 
            self.target_style == ParameterStyle.SQLITE):
            params = self._convert_array_params_for_sqlite(params)
        
        # Handle named parameters conversion
        if source_style == ParameterStyle.NAMED and self.target_style != ParameterStyle.NAMED:
            if isinstance(params, dict):
                # Convert named parameters to positional
                return tuple(params.values())
        
        # Ensure parameters are in the right format
        if isinstance(params, (list, tuple)):
            return params
        elif params is None:
            return ()
        else:
            return (params,)
    
    def adapt_query_and_params(self, query: str, params: Any = None, 
                             source_style: ParameterStyle = ParameterStyle.POSTGRESQL) -> Tuple[str, Any]:
        """
        Convert both query and parameters in one call.
        
        Args:
            query: SQL query string
            params: Parameter values
            source_style: Source parameter style
            
        Returns:
            Tuple of (converted_query, converted_params)
        """
        converted_query = self.convert_query(query, source_style)
        converted_params = self.convert_parameters(params, source_style)
        return converted_query, converted_params


# Factory function to create adapters for common database types
def create_query_adapter(db_provider_type: str) -> QueryAdapter:
    """
    Create a QueryAdapter for the given database provider type.
    
    Args:
        db_provider_type: Database provider type ('sqlite', 'postgresql', 'mysql', 'mssql')
        
    Returns:
        QueryAdapter instance
    """
    provider_type = db_provider_type.lower()
    
    if provider_type == 'sqlite':
        return QueryAdapter(ParameterStyle.SQLITE)
    elif provider_type in ('postgresql', 'postgres'):
        return QueryAdapter(ParameterStyle.POSTGRESQL)
    elif provider_type == 'mysql':
        return QueryAdapter(ParameterStyle.MYSQL)
    elif provider_type in ('mssql', 'sqlserver'):
        return QueryAdapter(ParameterStyle.MSSQL)
    else:
        # Default to PostgreSQL style for unknown providers
        return QueryAdapter(ParameterStyle.POSTGRESQL)


# Example usage:
if __name__ == "__main__":
    # Test the adapter
    sqlite_adapter = QueryAdapter(ParameterStyle.SQLITE)
    
    # Convert PostgreSQL query to SQLite
    pg_query = "SELECT * FROM users WHERE email = $1 AND active = $2"
    sqlite_query = sqlite_adapter.convert_query(pg_query, ParameterStyle.POSTGRESQL)
    print(f"PostgreSQL: {pg_query}")
    print(f"SQLite:     {sqlite_query}")
    
    # Convert parameters
    params = ("user@example.com", True)
    converted_params = sqlite_adapter.convert_parameters(params, ParameterStyle.POSTGRESQL)
    print(f"Parameters: {converted_params}")