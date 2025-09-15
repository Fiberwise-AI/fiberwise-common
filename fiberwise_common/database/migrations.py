"""
Migration management for FiberWise applications.
"""

import logging
from pathlib import Path
from typing import Optional, List
import asyncio

from .base import DatabaseProvider
from .sql_loader import load_sql_script

logger = logging.getLogger(__name__)

class MigrationManager:
    """Manages database migrations for FiberWise applications."""
    
    def __init__(self, db_provider: DatabaseProvider, migrations_dir: Optional[Path] = None):
        self.db_provider = db_provider
        self.migrations_dir = migrations_dir or self._get_default_migrations_dir()
    
    def _get_default_migrations_dir(self) -> Path:
        """Get the default migrations directory based on provider type."""
        # Try to find migrations directory relative to this module
        current_dir = Path(__file__).parent
        provider_name = self.db_provider.provider
        
        # Look for migrations in common directory structure
        possible_paths = [
            current_dir / "migrations" / provider_name,
            current_dir / "sql",  # Added: Look for sql directory where init.sql is located
            current_dir.parent / "migrations" / provider_name,
            current_dir.parent.parent / "database-scripts" / "migrations" / provider_name,
            current_dir.parent.parent / "database-scripts",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
                
        # Default fallback
        return current_dir / "migrations" / provider_name
    
    async def create_migrations_table(self):
        """Create the migrations tracking table."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
        await self.db_provider.execute(create_table_sql)
    
    async def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions."""
        try:
            rows = await self.db_provider.fetch_all(
                "SELECT version FROM schema_migrations ORDER BY version"
            )
            return [row['version'] if isinstance(row, dict) else row[0] for row in rows]
        except Exception:
            # Table doesn't exist yet
            return []
    
    async def mark_migration_applied(self, version: str):
        """Mark a migration as applied."""
        await self.db_provider.execute(
            "INSERT OR IGNORE INTO schema_migrations (version) VALUES (?)",
            version
        )
    
    async def apply_initial_schema(self):
        """Apply the initial database schema for the provider."""
        logger.info(f"Applying initial schema for {self.db_provider.provider}")
        
        try:
            # Load init.sql from packaged resources
            logger.info("Loading initial schema from packaged resources")
            schema = load_sql_script("init.sql")
            
            # Execute the schema
            await self._execute_sql_script(schema)
            
            # Mark as applied
            await self.mark_migration_applied("init")
            logger.info("Initial schema applied successfully")
            
        except FileNotFoundError:
            # Fallback to file system lookup for backward compatibility
            logger.warning("init.sql not found in package resources, trying file system")
            
            init_sql_paths = [
                self.migrations_dir / "init.sql",
                self.migrations_dir.parent / "init.sql",
                Path(__file__).parent.parent.parent / "database-scripts" / "init.sql"
            ]
            
            init_sql_path = None
            for path in init_sql_paths:
                if path.exists():
                    init_sql_path = path
                    break
            
            if not init_sql_path:
                logger.warning(f"No init.sql found for {self.db_provider.provider}")
                return
            
            logger.info(f"Loading initial schema from: {init_sql_path}")
            
            with open(init_sql_path, 'r') as f:
                schema = f.read()
            
            # Execute the schema
            await self._execute_sql_script(schema)
            
            # Mark as applied
            await self.mark_migration_applied("init")
            logger.info("Initial schema applied successfully")
            
        except Exception as e:
            logger.error(f"Failed to apply initial schema: {e}")
            raise
    
    
    async def _check_critical_tables(self) -> bool:
        """Check if critical tables exist in the database."""
        critical_tables = ['users', 'apps', 'app_versions', 'app_routes', 'agents']
        
        for table in critical_tables:
            try:
                # Check if table exists using PRAGMA table_info
                result = await self.db_provider.fetch_all(f"PRAGMA table_info({table})")
                if not result:
                    logger.info(f"=== MIGRATION DEBUG: Critical table '{table}' is missing ===")
                    return False
                else:
                    logger.info(f"=== MIGRATION DEBUG: Critical table '{table}' exists ===")
            except Exception as e:
                logger.info(f"=== MIGRATION DEBUG: Error checking table '{table}': {e} ===")
                return False
        
        logger.info(f"=== MIGRATION DEBUG: All critical tables exist ===")
        return True

    async def apply_migrations(self):
        """Apply all pending migrations."""
        logger.info("=== MIGRATION DEBUG: Starting apply_migrations ===")
        await self.create_migrations_table()
        
        applied_migrations = await self.get_applied_migrations()
        logger.info(f"=== MIGRATION DEBUG: Applied migrations found: {applied_migrations} ===")
        
        # Check if critical tables exist even if 'init' is marked as applied
        critical_tables_exist = await self._check_critical_tables()
        logger.info(f"=== MIGRATION DEBUG: Critical tables exist: {critical_tables_exist} ===")
        
        # If no migrations applied OR critical tables are missing, apply initial schema
        if not applied_migrations or not critical_tables_exist:
            if not applied_migrations:
                logger.info("=== MIGRATION DEBUG: No applied migrations, applying initial schema ===")
            else:
                logger.warning("=== MIGRATION DEBUG: Init marked as applied but critical tables missing - re-applying schema ===")
            
            try:
                await self.apply_initial_schema()
                # Refresh applied migrations list after initial schema
                applied_migrations = await self.get_applied_migrations()
                logger.info(f"=== MIGRATION DEBUG: After initial schema, applied migrations: {applied_migrations} ===")
            except Exception as e:
                logger.error(f"CRITICAL: Initial schema failed - {e}")
                raise Exception(f"Database initialization failed: {e}")
        else:
            logger.info("=== MIGRATION DEBUG: Migrations already applied and critical tables exist, skipping initial schema ===")
        
        # Look for migration files
        if not self.migrations_dir.exists():
            logger.info(f"No migrations directory found at {self.migrations_dir}")
            return
        
        migration_files = sorted(self.migrations_dir.glob("*.sql"))
        logger.info(f"=== MIGRATION DEBUG: Found migration files: {[f.name for f in migration_files]} ===")
        
        for migration_file in migration_files:
            version = migration_file.stem
            
            if version not in applied_migrations and version != "init":
                logger.info(f"Applying migration: {version}")
                
                try:
                    with open(migration_file, 'r') as f:
                        sql = f.read()
                    
                    await self._execute_sql_script(sql)
                    await self.mark_migration_applied(version)
                    
                    logger.info(f"Applied migration: {version}")
                    
                except Exception as e:
                    logger.error(f"CRITICAL: Migration {version} failed - {e}")
                    raise Exception(f"Migration {version} failed: {e}")
            else:
                logger.info(f"=== MIGRATION DEBUG: Skipping migration {version} (already applied or is init) ===")
        
        logger.info("=== MIGRATION DEBUG: Finished apply_migrations ===")
    
    async def _execute_sql_script(self, sql_script: str):
        """Execute a SQL script with multiple statements."""
        # DEBUG: Show first 1000 characters of raw SQL
        logger.info(f"=== RAW SQL FIRST 1000 CHARS ===")
        logger.info(f"{sql_script[:1000]}")
        logger.info(f"=== END RAW SQL ===")
        
        # Parse SQL statements properly, handling multi-line statements
        statements = self._parse_sql_statements(sql_script)
        
        logger.info(f"Executing {len(statements)} SQL statements")
        
        # Debug: Log first few statements with more detail
        for i, statement in enumerate(statements[:10]):
            statement_lines = statement.split('\n')
            first_line = statement_lines[0].strip()
            if len(statement_lines) > 1:
                second_line = statement_lines[1].strip() if len(statement_lines) > 1 else ""
                logger.info(f"=== STATEMENT {i+1} DEBUG: {first_line} | {second_line}... ===")
            else:
                logger.info(f"=== STATEMENT {i+1} DEBUG: {first_line}... ===")
        
        for i, statement in enumerate(statements):
            if statement:
                logger.debug(f"Executing statement {i+1}/{len(statements)}: {statement[:100]}...")
                try:
                    await self.db_provider.execute(statement)
                    logger.debug(f"Statement {i+1} executed successfully")
                except Exception as e:
                    logger.error(f"Failed to execute statement {i+1}: {e}")
                    logger.error(f"Statement: {statement}")
                    # Re-raise the exception to see what's actually failing
                    raise
    
    def _parse_sql_statements(self, sql_script: str) -> list:
        """Parse SQL script into individual statements, handling multi-line statements properly."""
        statements = []
        current_statement = ""
        in_string = False
        string_char = None
        paren_depth = 0
        
        logger.info(f"=== PARSING DEBUG: Starting to parse SQL script with {len(sql_script)} characters ===")
        
        i = 0
        while i < len(sql_script):
            char = sql_script[i]
            
            # Handle string literals
            if char in ('"', "'") and not in_string:
                in_string = True
                string_char = char
            elif char == string_char and in_string:
                # Check if it's escaped
                if i > 0 and sql_script[i-1] != '\\':
                    in_string = False
                    string_char = None
            
            # Handle parentheses (only when not in string)
            if not in_string:
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                elif char == ';' and paren_depth == 0:
                    # End of statement
                    statement = current_statement.strip()
                    if statement:
                        # Debug: Log each statement as it's parsed
                        first_words = ' '.join(statement.split()[:4])
                        logger.info(f"=== PARSED STATEMENT {len(statements)+1}: {first_words}... ===")
                        statements.append(statement)
                    current_statement = ""
                    i += 1
                    continue
            
            current_statement += char
            i += 1
        
        # Add the last statement if it doesn't end with semicolon
        if current_statement.strip():
            statement = current_statement.strip()
            first_words = ' '.join(statement.split()[:4])
            logger.info(f"=== PARSED FINAL STATEMENT {len(statements)+1}: {first_words}... ===")
            statements.append(statement)
        
        # Filter out empty statements and comments - fixed logic
        filtered_statements = []
        for stmt in statements:
            # Skip empty statements
            if not stmt:
                continue
            
            # Check if statement contains actual SQL (not just comments)
            # Split by lines and check if any line is not a comment
            lines = stmt.strip().split('\n')
            has_sql = False
            for line in lines:
                line = line.strip()
                if line and not line.startswith('--'):
                    has_sql = True
                    break
            
            # Skip comment-only statements
            if not has_sql:
                continue
                
            # Add valid statement
            filtered_statements.append(stmt)
        
        logger.info(f"=== PARSING DEBUG: Found {len(statements)} statements, filtered to {len(filtered_statements)} ===")
        
        # Debug: Show first few filtered statements
        for i, stmt in enumerate(filtered_statements[:5]):
            first_words = ' '.join(stmt.split()[:4])
            logger.info(f"=== FILTERED STATEMENT {i+1}: {first_words}... ===")
            
        # CRITICAL DEBUG: Show actual content of first few filtered statements
        for i, stmt in enumerate(filtered_statements[:3]):
            logger.info(f"=== FILTERED STATEMENT {i+1} FULL TEXT: {stmt[:200]}... ===")
            
        # SUPER CRITICAL: Check if users table got lost during parsing
        users_found = False
        for i, stmt in enumerate(filtered_statements):
            if 'CREATE TABLE IF NOT EXISTS users' in stmt:
                users_found = True
                logger.info(f"=== USERS TABLE FOUND at position {i+1} ===")
                break
        
        if not users_found:
            logger.error("=== CRITICAL: USERS TABLE NOT FOUND IN FILTERED STATEMENTS! ===")
            # Check if it was in the original parsed statements
            for i, stmt in enumerate(statements):
                if 'CREATE TABLE IF NOT EXISTS users' in stmt:
                    logger.error(f"=== USERS TABLE WAS IN ORIGINAL STATEMENTS at position {i+1} ===")
                    logger.error(f"=== USERS STATEMENT CONTENT: {stmt[:300]} ===")
                    break
        
        return filtered_statements
