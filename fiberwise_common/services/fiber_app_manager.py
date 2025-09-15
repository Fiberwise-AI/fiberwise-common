"""
App management services for install and update operations - Core business logic.
"""
import json
import hashlib
import zipfile
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import requests
from datetime import datetime

# Configuration constants
LOCAL_APP_INFO_FILENAME = "app.json"
DEFAULT_MANIFEST_FILENAMES = ["app_manifest.json", "app_manifest.yaml", "app_manifest.yml"]

# API endpoint paths
INSTALL_API_PATH = "/api/v1/deploy"
UPDATE_API_PATH_TEMPLATE = "/api/v1/updates/update/{app_id}"
UPLOAD_API_PATH_TEMPLATE = "/api/v1/deploy/{app_version_id}/upload"

# Configuration directories
FIBERWISE_DIR = Path.home() / ".fiberwise"
CONFIG_DIR = FIBERWISE_DIR / "configs"
DEFAULT_CONFIG_MARKER_FILE = FIBERWISE_DIR / "default_config.txt"


def get_default_instance_config() -> Optional[str]:
    """Get the name of the default instance configuration."""
    if DEFAULT_CONFIG_MARKER_FILE.exists():
        try:
            with open(DEFAULT_CONFIG_MARKER_FILE, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception:
            pass
    return None


def validate_instance_config(instance_name: Optional[str]) -> str:
    """
    Validate and return a valid instance configuration name.
    
    Args:
        instance_name: The instance name to validate, or None to use default
        
    Returns:
        str: Valid instance configuration name
        
    Raises:
        ValueError: If no valid configuration is found
    """
    if instance_name and instance_name.strip():
        return instance_name.strip()
    
    # Try to get default config
    default_config = get_default_instance_config()
    if default_config:
        # If default config is "default", update it to "local" for better UX
        if default_config == "default":
            try:
                with open(DEFAULT_CONFIG_MARKER_FILE, 'w', encoding='utf-8') as f:
                    f.write("local")
                return "local"
            except Exception:
                # If we can't update the file, still return "local"
                return "local"
        return default_config
    
    # Fallback to "local" as the default instance name
    return "local"


def load_instance_config(instance_name: str) -> Dict[str, Any]:
    """
    Load configuration for a specific instance.
    
    Args:
        instance_name: The name of the instance configuration to load
        
    Returns:
        Dict[str, Any]: Configuration data
        
    Raises:
        ValueError: If configuration cannot be loaded
    """
    if not instance_name or not instance_name.strip():
        raise ValueError("Instance name is required")
    
    # Sanitize config name for filename
    safe_filename = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in instance_name)
    config_file = CONFIG_DIR / f"{safe_filename}.json"
    
    if not config_file.exists():
        raise ValueError(f"Configuration '{instance_name}' not found. Use 'fiber account add-config' to create it.")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Validate required fields
        api_key = config_data.get('api_key') or config_data.get('fiberwise_api_key')
        base_url = config_data.get('base_url') or config_data.get('fiberwise_base_url')
        
        if not api_key or not base_url:
            raise ValueError(f"Configuration '{instance_name}' is missing required fields (api_key, base_url)")
        
        return {
            'api_key': api_key,
            'base_url': base_url,
            'instance_name': instance_name
        }
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration '{instance_name}': {e}")
    except Exception as e:
        raise ValueError(f"Error loading configuration '{instance_name}': {e}")

class AppOperationResult:
    """Result of an app operation with status and messages."""
    def __init__(self, success: bool, message: str = "", data: Dict[str, Any] = None):
        self.success = success
        self.message = message
        self.data = data or {}
        self.warnings: List[str] = []
        self.info_messages: List[str] = []
    
    def add_warning(self, warning: str):
        self.warnings.append(warning)
    
    def add_info(self, info: str):
        self.info_messages.append(info)

class FiberAppManager:
    """Core business logic for app installation and update operations."""
    
    def __init__(self, api_endpoint: str, api_key: str, instance_name: str):
        """
        Initialize FiberAppManager with required instance configuration.
        
        Args:
            api_endpoint: The API endpoint URL
            api_key: The API key for authentication
            instance_name: The instance name (required) - determines where app info is stored
        """
        if not instance_name or not instance_name.strip():
            raise ValueError("instance_name is required and cannot be empty")
        
        self.api_endpoint = api_endpoint.rstrip('/')
        self.api_key = api_key
        self.instance_name = instance_name.strip()
        
        # Ensure API key has the correct prefix for server validation
        if not self.api_key.startswith('api_'):
            self.api_key = f"api_{self.api_key}"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    @classmethod
    def from_instance_config(cls, instance_name: Optional[str] = None) -> 'FiberAppManager':
        """
        Create FiberAppManager from instance configuration.
        
        Args:
            instance_name: Instance name, or None to use default
            
        Returns:
            FiberAppManager: Configured instance
            
        Raises:
            ValueError: If configuration is invalid or missing
        """
        validated_instance = validate_instance_config(instance_name)
        config = load_instance_config(validated_instance)
        
        return cls(
            api_endpoint=config['base_url'],
            api_key=config['api_key'],
            instance_name=validated_instance
        )

    def install_app(self, app_path: Path, manifest_path: Optional[Path] = None) -> AppOperationResult:
        """Install an app from a local directory."""
        try:
            # Determine manifest path
            if not manifest_path:
                manifest_path = self._find_default_manifest(app_path)
                if not manifest_path:
                    return AppOperationResult(False, "No manifest file found. Please specify with --manifest or create app_manifest.json")
            
            # Load and parse manifest
            manifest_data = self._load_and_parse_manifest(manifest_path)
            if not manifest_data:
                return AppOperationResult(False, "Failed to load or parse manifest file")

            # Validate manifest
            validation_result = self._validate_and_normalize_manifest(manifest_data)
            if not validation_result.success:
                return validation_result

            validated_manifest = validation_result.data
            
            # Validate that function implementation files exist
            function_validation_result = self._validate_function_files(app_path, validated_manifest)
            if not function_validation_result.success:
                return function_validation_result
            
            # NEW: Validate that pipeline step implementation files exist  
            pipeline_validation_result = self._validate_pipeline_files(app_path, validated_manifest)
            if not pipeline_validation_result.success:
                return pipeline_validation_result
            
            # IMPORTANT: Create temporary app info BEFORE building to prevent Vite config errors
            # The build process needs this file to read app metadata
            print(f"[DEBUG] Creating temporary app info for build...")
            temp_app_info = {
                "app_id": validated_manifest.get("app_id", "unknown"),
                "version_id": validated_manifest.get("version", "latest"),
                "temp": True  # Mark as temporary
            }
            temp_save_result = self._save_local_app_info(app_path, temp_app_info)
            if not temp_save_result.success:
                print(f"[WARNING] Failed to create temporary app info: {temp_save_result.message}")
            
            # IMPORTANT: Build the app BEFORE sending manifest to prevent "fake updates"
            # This ensures the dist folder is fresh and any build-time changes are captured
            print(f"[DEBUG] Building app before installation...")
            build_result = self._build_app(app_path)
            if not build_result.success:
                return AppOperationResult(False, f"Build failed: {build_result.message}")
            
            # Step 1: Send manifest to backend
            print(f"[DEBUG] Sending manifest to backend: {self.api_endpoint}{INSTALL_API_PATH}")
            response = requests.post(
                f"{self.api_endpoint}{INSTALL_API_PATH}",
                headers=self.headers,
                json=validated_manifest,
                timeout=30
            )
            print(f"[DEBUG] Backend response: {response.status_code}")
            
            if response.status_code not in (200, 201):
                return AppOperationResult(False, f"Server error {response.status_code}: {response.text}")
            
            install_response = response.json()
            print(f"[DEBUG] Install response received, processing results...")
            result = AppOperationResult(True, "Manifest processed successfully")
            result.data["install_response"] = install_response
            
            # Process app results
            app_results = install_response.get("app", [])
            if app_results:
                app_result = app_results[0]
                if app_result.get("success"):
                    result.add_info(f"App '{app_result.get('name', 'Unknown')}' installed successfully")
                    result.data["app_version_id"] = app_result.get("version_id")
                    print(f"[DEBUG] App created with version_id: {app_result.get('version_id')}")
                else:
                    error_msg = app_result.get("error", "")
                    if "already exists" in error_msg.lower():
                        result.success = False
                        result.message = "App already exists. Use update instead."
                        result.data["suggestion"] = "update"
                        return result
                    else:
                        result.add_warning(f"App installation failed: {app_result.get('message', 'Unknown error')}")
            
            # Process agent results
            agent_results = install_response.get("agents", [])
            successful_agents = [a for a in agent_results if a.get("success")]
            failed_agents = [a for a in agent_results if not a.get("success")]
            
            for agent in successful_agents:
                result.add_info(f"Agent '{agent.get('name', 'Unknown')}' installed successfully")
            
            for agent in failed_agents:
                result.add_warning(f"Agent '{agent.get('name', 'Unknown')}' failed: {agent.get('message', 'Unknown error')}")
            
            # Step 2: Handle file upload if we have app_version_id
            app_version_id = result.data.get("app_version_id")
            if app_version_id:
                print(f"[DEBUG] Starting bundle creation and upload for app_version_id: {app_version_id}")
                bundle_result = self._create_and_upload_bundle(app_path, app_version_id)
                if bundle_result.success:
                    result.add_info("App bundle uploaded successfully")
                    print(f"[DEBUG] Bundle upload completed successfully")
                else:
                    result.add_warning(f"Bundle upload failed: {bundle_result.message}")
                    print(f"[DEBUG] Bundle upload failed: {bundle_result.message}")
                
                # Save local app info
                print(f"[DEBUG] Saving local app info...")
                app_info = {
                    "app_id": app_results[0].get("component_id") or app_results[0].get("app_id") if app_results else None,
                    "app_version_id": app_version_id,
                    "last_manifest_version": self._get_app_version(validated_manifest),
                    "last_manifest_hash": self._calculate_manifest_hash(manifest_path),
                    "last_python_files_hash": self._calculate_python_files_hash(app_path, validated_manifest),
                    "current_app_dir": str(app_path.resolve()),
                    "manifest_path": str(manifest_path.resolve())
                }
                
                save_result = self._save_local_app_info(app_path, app_info)
                if not save_result.success:
                    result.add_warning(f"Failed to save local app info: {save_result.message}")
                else:
                    print(f"[DEBUG] Local app info saved successfully")
            else:
                print(f"[DEBUG] No app_version_id found, skipping bundle upload")
            
            # Final success determination
            if not app_version_id and not successful_agents:
                result.success = False
                result.message = "No components were successfully processed"
            
            # Step 3: Process OAuth authenticators from .fiber/{instance_name}/oauth/ directory
            oauth_result = self._process_oauth_authenticators(app_path, validated_manifest)
            if oauth_result.warnings:
                for warning in oauth_result.warnings:
                    result.add_warning(warning)
            if oauth_result.info_messages:
                for info in oauth_result.info_messages:
                    result.add_info(info)
            
            return result
            
        except requests.RequestException as e:
            return AppOperationResult(False, f"Network error: {e}")
        except Exception as e:
            return AppOperationResult(False, f"Unexpected error: {e}")

    def update_app(self, app_path: Path, manifest_path: Optional[Path] = None, force: bool = False) -> AppOperationResult:
        """Update an existing app."""
        try:
            # Check if path has changed
            path_changed, previous_path = self._check_path_changed(app_path)
            if path_changed and not force:
                return AppOperationResult(False, f"App directory has moved from {previous_path} to {app_path}. Use --force to update anyway.")
            
            # Load local app info
            app_info = self._load_local_app_info(app_path)
            if not app_info:
                return AppOperationResult(False, "No local app info found. Use install command first.")
            
            app_id = app_info.get("app_id")
            if not app_id:
                return AppOperationResult(False, "Local app info missing app_id")
            
            # Determine manifest path
            if not manifest_path:
                stored_manifest_path = app_info.get("manifest_path")
                if stored_manifest_path and Path(stored_manifest_path).exists():
                    manifest_path = Path(stored_manifest_path)
                else:
                    manifest_path = self._find_default_manifest(app_path)
                
                if not manifest_path:
                    return AppOperationResult(False, "No manifest file found. Please specify with --manifest")
            
            # Load and validate manifest
            manifest_data = self._load_and_parse_manifest(manifest_path)
            if not manifest_data:
                return AppOperationResult(False, "Failed to load or parse manifest file")
            
            validation_result = self._validate_and_normalize_manifest(manifest_data)
            if not validation_result.success:
                return validation_result

            validated_manifest = validation_result.data
            
            # Validate that function implementation files exist
            function_validation_result = self._validate_function_files(app_path, validated_manifest)
            if not function_validation_result.success:
                return function_validation_result
            
            # IMPORTANT: Build the app BEFORE hash comparison to ensure fresh build
            # This prevents "fake updates" where only source changed but build is stale
            print(f"[DEBUG] Building app before update check...")
            build_result = self._build_app(app_path)
            if not build_result.success:
                return AppOperationResult(False, f"Build failed: {build_result.message}")
            
            # Check if update is needed (after build, so we compare fresh dist folder)
            if not force:
                current_manifest_hash = self._calculate_manifest_hash(manifest_path)
                last_manifest_hash = app_info.get("last_manifest_hash")
                
                current_python_hash = self._calculate_python_files_hash(app_path, validated_manifest)
                last_python_hash = app_info.get("last_python_files_hash")
                
                current_version = self._get_app_version(validated_manifest)
                last_version = app_info.get("last_manifest_version")
                
                # Check if anything changed: manifest, Python files, or version
                manifest_unchanged = current_manifest_hash and last_manifest_hash and current_manifest_hash == last_manifest_hash
                python_unchanged = current_python_hash and last_python_hash and current_python_hash == last_python_hash
                version_unchanged = current_version == last_version
                
                if manifest_unchanged and python_unchanged and version_unchanged:
                    return AppOperationResult(True, "No changes detected. App is already up to date.")            # Step 1: Send update request (using Form data as expected by API)
            import json
            form_data = {
                'manifest': json.dumps(validated_manifest)
            }
            
            response = requests.post(
                f"{self.api_endpoint}{UPDATE_API_PATH_TEMPLATE.format(app_id=app_id)}",
                headers={k: v for k, v in self.headers.items() if k != 'Content-Type'},  # Remove JSON content-type
                data=form_data,  # Use form data instead of json
                timeout=30
            )
            
            if response.status_code not in (200, 201):
                return AppOperationResult(False, f"Server error {response.status_code}: {response.text}")
            
            update_response = response.json()
            app_version_id = update_response.get("app_version_id")
            
            if not app_version_id:
                return AppOperationResult(False, "No app_version_id returned from server")
            
            result = AppOperationResult(True, "App updated successfully")
            result.data["update_response"] = update_response
            result.data["app_version_id"] = app_version_id
            
            # Step 2: Create and upload app bundle
            bundle_result = self._create_and_upload_bundle(app_path, app_version_id)
            if not bundle_result.success:
                return AppOperationResult(False, f"Bundle upload failed: {bundle_result.message}")
            
            # Step 3: Update local app info
            app_info_updates = {
                "app_version_id": app_version_id,
                "last_manifest_version": self._get_app_version(validated_manifest),
                "last_manifest_hash": self._calculate_manifest_hash(manifest_path),
                "last_python_files_hash": self._calculate_python_files_hash(app_path, validated_manifest),
                "current_app_dir": str(app_path.resolve()),
                "manifest_path": str(manifest_path.resolve())
            }
            
            existing_app_info = self._load_local_app_info(app_path) or {}
            existing_app_info.update(app_info_updates)
            
            save_result = self._save_local_app_info(app_path, existing_app_info)
            if not save_result.success:
                result.add_warning(f"Failed to save local app info: {save_result.message}")
            
            app_name = self._get_app_name(validated_manifest)
            result.message = f"App '{app_name}' updated successfully"
            
            return result
            
        except requests.RequestException as e:
            return AppOperationResult(False, f"Network error: {e}")
        except Exception as e:
            return AppOperationResult(False, f"Unexpected error: {e}")

    def _find_default_manifest(self, search_dir: Path) -> Optional[Path]:
        """Looks for default manifest files in the specified directory."""
        for filename in DEFAULT_MANIFEST_FILENAMES:
            path = search_dir / filename
            if path.is_file():
                return path
        return None

    def _load_and_parse_manifest(self, manifest_path: Path) -> Optional[Dict[str, Any]]:
        """Load and parse manifest file (JSON or YAML)."""
        try:
            content = manifest_path.read_text(encoding='utf-8')
            
            if manifest_path.suffix.lower() == '.json':
                return json.loads(content)
            else:
                return yaml.safe_load(content)
        except Exception:
            return None

    def _calculate_manifest_hash(self, manifest_path: Path) -> Optional[str]:
        """Calculate SHA256 hash of manifest file content."""
        try:
            content = manifest_path.read_text(encoding='utf-8')
            return hashlib.sha256(content.encode('utf-8')).hexdigest()
        except Exception:
            return None
    
    def _calculate_python_files_hash(self, app_path: Path, manifest_data: Dict[str, Any]) -> Optional[str]:
        """Calculate combined hash of all Python function and agent files."""
        try:
            python_files = []
            
            # Get function files
            functions = manifest_data.get('functions', [])
            for func in functions:
                impl_path = func.get('implementation_path')
                if impl_path:
                    full_path = app_path / impl_path
                    if full_path.exists():
                        python_files.append(full_path)
            
            # Get agent files
            agents = manifest_data.get('agents', [])
            for agent in agents:
                impl_path = agent.get('implementation_path')
                if impl_path:
                    full_path = app_path / impl_path
                    if full_path.exists():
                        python_files.append(full_path)
            
            # Get pipeline step files
            pipelines = manifest_data.get('pipelines', [])
            for pipeline in pipelines:
                structure = pipeline.get('structure', {})
                steps = structure.get('steps', [])
                for step in steps:
                    impl_path = step.get('implementation_path')
                    if impl_path:
                        full_path = app_path / impl_path
                        if full_path.exists():
                            python_files.append(full_path)
            
            if not python_files:
                return None
            
            # Sort files for consistent hashing
            python_files.sort(key=lambda p: str(p))
            
            # Calculate combined hash
            hasher = hashlib.sha256()
            for file_path in python_files:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    hasher.update(f"{file_path.name}:{content}".encode('utf-8'))
                except Exception:
                    # If we can't read a file, include its modification time
                    hasher.update(f"{file_path.name}:{file_path.stat().st_mtime}".encode('utf-8'))
            
            return hasher.hexdigest()
        except Exception:
            return None

    def _validate_and_normalize_manifest(self, manifest_data: Dict[str, Any]) -> AppOperationResult:
        """Validate and normalize manifest data to handle both old and new formats."""
        try:
            # Legacy validation - check for required fields and normalize
            normalized = dict(manifest_data)
            
            # Handle old-style flat structure
            if "app_name" in normalized or "app_version" in normalized:
                # Convert to new nested structure
                app_data = {
                    "name": normalized.get("app_name", normalized.get("name", "Unknown")),
                    "version": normalized.get("app_version", normalized.get("version", "1.0.0")),
                    "description": normalized.get("description", ""),
                    "app_slug": normalized.get("app_slug", normalized.get("app_name", "").lower().replace(" ", "-"))
                }
                
                # Remove old fields and add app object
                for old_field in ["app_name", "app_version"]:
                    normalized.pop(old_field, None)
                
                normalized["app"] = app_data
            
            # Ensure we have required app fields
            if "app" not in normalized:
                return AppOperationResult(False, "Manifest missing 'app' section")
            
            app_section = normalized["app"]
            required_app_fields = ["name", "version"]
            missing_fields = [field for field in required_app_fields if not app_section.get(field)]
            
            if missing_fields:
                return AppOperationResult(False, f"App section missing required fields: {missing_fields}")
            
            result = AppOperationResult(True, "Manifest validation successful")
            result.data = normalized
            return result
            
        except Exception as e:
            return AppOperationResult(False, f"Error validating manifest: {e}")

    def _get_app_name(self, manifest_data: Dict[str, Any]) -> str:
        """Extract app name from normalized manifest."""
        if "app" in manifest_data:
            return manifest_data["app"].get("name", "Unknown")
        return manifest_data.get("app_name", manifest_data.get("name", "Unknown"))
    
    def _get_app_version(self, manifest_data: Dict[str, Any]) -> str:
        """Extract app version from normalized manifest."""
        if "app" in manifest_data:
            return manifest_data["app"].get("version", "1.0.0")
        return manifest_data.get("app_version", manifest_data.get("version", "1.0.0"))

    def _validate_function_files(self, app_path: Path, manifest_data: Dict[str, Any]) -> AppOperationResult:
        """Validate that function implementation files referenced in manifest actually exist."""
        try:
            functions = manifest_data.get("functions", [])
            if not functions:
                return AppOperationResult(True, "No functions to validate")
            
            missing_files = []
            for func in functions:
                func_name = func.get("name", "Unknown")
                implementation_path = func.get("implementation_path")
                
                if implementation_path:
                    full_path = app_path / implementation_path
                    if not full_path.exists():
                        missing_files.append(f"Function '{func_name}': {implementation_path}")
                        print(f"[DEBUG] Missing function file: {full_path}")
                    else:
                        print(f"[DEBUG] Found function file: {full_path}")
            
            if missing_files:
                error_msg = f"Missing function implementation files:\n" + "\n".join([f"  - {f}" for f in missing_files])
                return AppOperationResult(False, error_msg)
            
            print(f"[DEBUG] All {len(functions)} function files validated successfully")
            return AppOperationResult(True, f"All function implementation files found ({len(functions)} functions)")
            
        except Exception as e:
            return AppOperationResult(False, f"Error validating function files: {e}")

    def _validate_pipeline_files(self, app_path: Path, manifest_data: Dict[str, Any]) -> AppOperationResult:
        """Validate that pipeline step implementation files referenced in manifest actually exist."""
        try:
            pipelines = manifest_data.get("pipelines", [])
            if not pipelines:
                return AppOperationResult(True, "No pipelines to validate")
            
            missing_files = []
            validation_errors = []
            total_steps = 0
            
            for i, pipeline in enumerate(pipelines):
                if not isinstance(pipeline, dict):
                    validation_errors.append(f"Pipeline at index {i} is not a valid object.")
                    continue

                pipeline_name = pipeline.get("name")
                if not pipeline_name:
                    validation_errors.append(f"Pipeline at index {i} is missing a 'name'.")
                    continue

                structure = pipeline.get("structure")
                if not structure or not isinstance(structure, dict):
                    validation_errors.append(f"Pipeline '{pipeline_name}' is missing a valid 'structure' object.")
                    continue
                
                steps = structure.get("steps", [])
                
                for j, step in enumerate(steps):
                    total_steps += 1
                    if not isinstance(step, dict):
                        validation_errors.append(f"Pipeline '{pipeline_name}' step at index {j} is not a valid object.")
                        continue

                    step_id = step.get("id")
                    if not step_id:
                        validation_errors.append(f"Pipeline '{pipeline_name}' step at index {j} is missing an 'id'.")
                        continue

                    # Validate step_class (required for graph-based pipelines)
                    step_class = step.get("step_class")
                    
                    if not step_class:
                        validation_errors.append(f"Pipeline '{pipeline_name}' step '{step_id}' missing required 'step_class'.")
                        continue
                    
                    # Validate that step class exists in pipelines directory
                    pipeline_dir = app_path / "pipelines"
                    if pipeline_dir.exists():
                        # Search for the step class in Python files
                        step_class_found = False
                        for py_file in pipeline_dir.rglob("*.py"):
                            try:
                                with open(py_file, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    if f"class {step_class}" in content:
                                        step_class_found = True
                                        break
                            except Exception:
                                continue
                        
                        if not step_class_found:
                            missing_files.append(f"Pipeline '{pipeline_name}' step '{step_id}': step_class '{step_class}' not found in pipelines/ directory")
                    else:
                        missing_files.append(f"Pipeline '{pipeline_name}': pipelines/ directory not found")
            
            if validation_errors:
                error_msg = "Manifest validation errors for pipelines:\n" + "\n".join([f"  - {e}" for e in validation_errors])
                return AppOperationResult(False, error_msg)

            if missing_files:
                error_msg = f"Missing pipeline step implementation files:\n" + "\n".join([f"  - {f}" for f in missing_files])
                return AppOperationResult(False, error_msg)
            
            print(f"[DEBUG] All {total_steps} pipeline step files validated successfully")
            return AppOperationResult(True, f"All pipeline step implementation files found ({total_steps} steps)")
            
        except Exception as e:
            return AppOperationResult(False, f"Error validating pipeline files: {e}")

    def _create_and_upload_bundle(self, app_path: Path, app_version_id: str) -> AppOperationResult:
        """Create and upload app bundle (assumes app is already built)."""
        print(f"[DEBUG] Creating bundle for app at: {app_path}")
        print(f"[DEBUG] App version ID: {app_version_id}")
        
        # Create bundle (no need to build again, already done in install_app)
        bundle_result = self._create_app_bundle(app_path)
        if not bundle_result.success:
            print(f"[DEBUG] Bundle creation failed: {bundle_result.message}")
            return bundle_result
        
        bundle_path = Path(bundle_result.data["bundle_path"])
        print(f"[DEBUG] Bundle created successfully: {bundle_path}")
        
        try:
            # Step 2: Upload bundle
            upload_result = self._upload_bundle(app_version_id, bundle_path)
            print(f"[DEBUG] Upload result: {upload_result.success} - {upload_result.message}")
            return upload_result
        finally:
            # Clean up temporary bundle
            if bundle_path.exists():
                bundle_path.unlink()
                print(f"[DEBUG] Cleaned up bundle file: {bundle_path}")

    def _build_app(self, app_path: Path) -> AppOperationResult:
        """Build the app if it has a package.json with build script."""
        import subprocess
        import json
        
        package_json_path = app_path / "package.json"
        
        # Check if package.json exists
        if not package_json_path.exists():
            print(f"[DEBUG] No package.json found at {package_json_path}, skipping build")
            return AppOperationResult(True, "No build needed (no package.json)")
        
        try:
            # Read package.json to check for build script
            with open(package_json_path, 'r') as f:
                package_data = json.load(f)
            
            scripts = package_data.get('scripts', {})
            if 'build' not in scripts:
                print(f"[DEBUG] No build script found in package.json, skipping build")
                return AppOperationResult(True, "No build needed (no build script)")
            
            print(f"[DEBUG] Found build script: {scripts['build']}")
            
            # Check if node_modules exists, if not run npm install first
            node_modules_path = app_path / "node_modules"
            if not node_modules_path.exists():
                print(f"[DEBUG] node_modules not found, running npm install first...")
                
                # Try different npm commands for install (Windows compatibility)
                npm_commands = ['npm', 'npm.cmd', 'npm.exe']
                install_result = None
                
                for npm_cmd in npm_commands:
                    try:
                        print(f"[DEBUG] Trying {npm_cmd} install command...")
                        install_result = subprocess.run(
                            [npm_cmd, 'install'],
                            cwd=app_path,
                            capture_output=True,
                            text=True,
                            timeout=300  # 5 minute timeout
                        )
                        print(f"[DEBUG] {npm_cmd} install command worked!")
                        break
                    except FileNotFoundError:
                        print(f"[DEBUG] {npm_cmd} not found for install, trying next...")
                        continue
                
                if install_result is None:
                    # Try using PowerShell as a fallback (Windows)
                    try:
                        print(f"[DEBUG] Trying PowerShell npm install command...")
                        install_result = subprocess.run(
                            ['powershell', '-Command', 'npm install'],
                            cwd=app_path,
                            capture_output=True,
                            text=True,
                            timeout=300
                        )
                        print(f"[DEBUG] PowerShell npm install command worked!")
                    except FileNotFoundError:
                        print(f"[DEBUG] PowerShell not found for install either")
                        
                if install_result is None:
                    return AppOperationResult(False, "npm command not found for install. Please ensure Node.js and npm are installed and in PATH.")
                
                if install_result.returncode != 0:
                    print(f"[DEBUG] npm install failed with exit code {install_result.returncode}")
                    print(f"[DEBUG] install stderr: {install_result.stderr}")
                    return AppOperationResult(False, f"npm install failed: {install_result.stderr}")
                
                print(f"[DEBUG] npm install completed successfully")
                print(f"[DEBUG] Install output: {install_result.stdout}")
            else:
                print(f"[DEBUG] node_modules exists, skipping npm install")
            
            print(f"[DEBUG] Running npm run build in {app_path}")
            
            # Try different npm commands (Windows compatibility)
            npm_commands = ['npm', 'npm.cmd', 'npm.exe']
            result = None
            
            for npm_cmd in npm_commands:
                try:
                    print(f"[DEBUG] Trying {npm_cmd} command...")
                    result = subprocess.run(
                        [npm_cmd, 'run', 'build'],
                        cwd=app_path,
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minute timeout
                    )
                    print(f"[DEBUG] {npm_cmd} command worked!")
                    break
                except FileNotFoundError:
                    print(f"[DEBUG] {npm_cmd} not found, trying next...")
                    continue
            
            if result is None:
                # Try using PowerShell as a fallback (Windows)
                try:
                    print(f"[DEBUG] Trying PowerShell npm command...")
                    result = subprocess.run(
                        ['powershell', '-Command', 'npm run build'],
                        cwd=app_path,
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    print(f"[DEBUG] PowerShell npm command worked!")
                except FileNotFoundError:
                    print(f"[DEBUG] PowerShell not found either")
                    
            if result is None:
                return AppOperationResult(False, "npm command not found. Please ensure Node.js and npm are installed and in PATH.")
            
            if result.returncode != 0:
                print(f"[DEBUG] Build failed with exit code {result.returncode}")
                print(f"[DEBUG] stderr: {result.stderr}")
                return AppOperationResult(False, f"Build failed: {result.stderr}")
            
            print(f"[DEBUG] Build completed successfully")
            print(f"[DEBUG] Build output: {result.stdout}")
            
            # Check if dist directory was created
            dist_path = app_path / "dist"
            if not dist_path.exists():
                return AppOperationResult(False, "Build completed but no dist folder was created")
            
            return AppOperationResult(True, "App built successfully")
            
        except json.JSONDecodeError as e:
            return AppOperationResult(False, f"Invalid package.json: {e}")
        except subprocess.TimeoutExpired:
            return AppOperationResult(False, "Build timed out (took more than 5 minutes)")
        except Exception as e:
            return AppOperationResult(False, f"Build error: {e}")

    def _create_app_bundle(self, app_path: Path) -> AppOperationResult:
        """Create a ZIP bundle of the app directory with manifest and agents always included."""
        try:
            bundle_path = app_path / f"{app_path.name}_bundle.zip"
            print(f"[DEBUG] Creating bundle at: {bundle_path}")
            
            files_added = 0
            excluded_files = 0
            
            # Check if dist folder exists (prioritize it for web apps)
            dist_path = app_path / "dist"
            
            with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # ALWAYS add the manifest file first (regardless of dist folder existence)
                manifest_files = ["app_manifest.yaml", "app_manifest.yml", "manifest.yaml", "manifest.yml"]
                for manifest_file in manifest_files:
                    manifest_path = app_path / manifest_file
                    if manifest_path.exists():
                        zipf.write(manifest_path, manifest_file)
                        files_added += 1
                        print(f"[DEBUG] Added manifest: {manifest_file}")
                        break
                
                # ALWAYS add agent implementations if they exist (regardless of dist folder existence)
                agents_dir = app_path / "agents"
                if agents_dir.exists() and agents_dir.is_dir():
                    def walk_agents_directory(current_path, base_path):
                        nonlocal files_added
                        try:
                            for item in current_path.iterdir():
                                if item.is_file() and not item.name.endswith('.pyc'):
                                    arcname = item.relative_to(base_path)
                                    zipf.write(item, arcname)
                                    files_added += 1
                                    print(f"[DEBUG] Added agent file: {arcname}")
                                elif item.is_dir():
                                    walk_agents_directory(item, base_path)
                        except (OSError, PermissionError) as e:
                            print(f"[DEBUG] Warning: Could not read agents directory {current_path}: {e}")
                    
                    walk_agents_directory(agents_dir, app_path)
                    print(f"[DEBUG] Finished adding agents directory")
                
                # ALWAYS add function implementations if they exist (regardless of dist folder existence)
                functions_dir = app_path / "functions"
                if functions_dir.exists() and functions_dir.is_dir():
                    def walk_functions_directory(current_path, base_path):
                        nonlocal files_added
                        try:
                            for item in current_path.iterdir():
                                if item.is_file() and not item.name.endswith('.pyc'):
                                    arcname = item.relative_to(base_path)
                                    zipf.write(item, arcname)
                                    files_added += 1
                                    print(f"[DEBUG] Added function file: {arcname}")
                                elif item.is_dir():
                                    walk_functions_directory(item, base_path)
                        except (OSError, PermissionError) as e:
                            print(f"[DEBUG] Warning: Could not read functions directory {current_path}: {e}")
                    
                    walk_functions_directory(functions_dir, app_path)
                    print(f"[DEBUG] Finished adding functions directory")
                
                # ALWAYS add pipeline implementations if they exist (regardless of dist folder existence)
                pipelines_dir = app_path / "pipelines"
                if pipelines_dir.exists() and pipelines_dir.is_dir():
                    def walk_pipelines_directory(current_path, base_path):
                        nonlocal files_added
                        try:
                            for item in current_path.iterdir():
                                if item.is_file() and not item.name.endswith('.pyc'):
                                    arcname = item.relative_to(base_path)
                                    zipf.write(item, arcname)
                                    files_added += 1
                                    print(f"[DEBUG] Added pipeline file: {arcname}")
                                elif item.is_dir():
                                    walk_pipelines_directory(item, base_path)
                        except (OSError, PermissionError) as e:
                            print(f"[DEBUG] Warning: Could not read pipelines directory {current_path}: {e}")
                    
                    walk_pipelines_directory(pipelines_dir, app_path)
                    print(f"[DEBUG] Finished adding pipelines directory")
                
                if dist_path.exists() and dist_path.is_dir():
                    # Bundle the contents of the dist folder for the built app
                    print(f"[DEBUG] Found dist folder, bundling contents from: {dist_path}")
                    def walk_dist_directory(current_path, base_path):
                        nonlocal files_added, excluded_files
                        try:
                            items = list(current_path.iterdir())
                            print(f"[DEBUG] Processing dist directory {current_path} with {len(items)} items")
                            
                            for item in items:
                                try:
                                    if item.is_file():
                                        arcname = item.relative_to(base_path)
                                        zipf.write(item, arcname)
                                        files_added += 1
                                        if files_added % 50 == 0:  # Progress indicator
                                            print(f"[DEBUG] Added {files_added} files...")
                                    elif item.is_dir():
                                        walk_dist_directory(item, base_path)
                                except (OSError, PermissionError) as e:
                                    print(f"[DEBUG] Warning: Could not access {item}: {e}")
                                    continue
                        except (OSError, PermissionError) as e:
                            print(f"[DEBUG] Warning: Could not read directory {current_path}: {e}")
                            return
                    
                    walk_dist_directory(dist_path, dist_path)
                else:
                    # No dist folder - bundle the entire app directory but exclude common build artifacts
                    print(f"[DEBUG] No dist folder found, bundling entire app directory excluding build artifacts")
                    def walk_directory(current_path, base_path):
                        nonlocal files_added, excluded_files
                        try:
                            items = list(current_path.iterdir())
                            print(f"[DEBUG] Processing directory {current_path} with {len(items)} items")
                            
                            for item in items:
                                try:
                                    if item.is_file():
                                        # Skip manifest files since we already added them
                                        if item.name in manifest_files:
                                            continue
                                        if not self._should_exclude_file(base_path, item):
                                            arcname = item.relative_to(base_path)
                                            zipf.write(item, arcname)
                                            files_added += 1
                                            if files_added % 50 == 0:  # Progress indicator
                                                print(f"[DEBUG] Added {files_added} files...")
                                        else:
                                            excluded_files += 1
                                    elif item.is_dir():
                                        # Skip agents and functions directories since we already added them
                                        if item.name in ["agents", "functions", "pipelines"]:
                                            continue
                                        # Check if directory should be excluded before recursing
                                        if not self._should_exclude_directory(base_path, item):
                                            walk_directory(item, base_path)
                                        else:
                                            print(f"[DEBUG] Excluding directory: {item}")
                                except (OSError, PermissionError) as e:
                                    print(f"[DEBUG] Warning: Could not access {item}: {e}")
                                    continue
                        except (OSError, PermissionError) as e:
                            print(f"[DEBUG] Warning: Could not read directory {current_path}: {e}")
                            return
                    
                    walk_directory(app_path, app_path)
            
            print(f"[DEBUG] Bundle creation complete: {files_added} files added, {excluded_files} excluded")
            result = AppOperationResult(True, f"Bundle created with {files_added} files")
            result.data["bundle_path"] = str(bundle_path)
            result.data["files_count"] = files_added
            return result
            
        except Exception as e:
            print(f"[DEBUG] Bundle creation failed: {e}")
            return AppOperationResult(False, f"Error creating app bundle: {e}")

    def _should_exclude_file(self, app_path: Path, file_path: Path) -> bool:
        """Determine if a file should be excluded from the bundle."""
        relative_path = file_path.relative_to(app_path)
        path_parts = relative_path.parts
        
        # Exclude directory patterns (check if any part of the path matches these)
        exclude_dir_patterns = [
            '.git', '__pycache__', 'node_modules', '.venv', 'venv', 
            '.fiber', '.pytest_cache', '.coverage', '.mypy_cache',
            'dist', 'build', 'egg-info'
        ]
        
        for pattern in exclude_dir_patterns:
            if pattern in path_parts:
                return True
        
        # Exclude file patterns
        exclude_file_patterns = [
            '.gitignore', '.DS_Store', 'Thumbs.db',
            '.env', '.env.local', '.env.production',
            '*.pyc', '*.pyo', '*.pyd', '*.so', '*.dylib', '*.dll',
            '*.log', '*.tmp', '*_bundle.zip'
        ]
        
        file_name = file_path.name
        for pattern in exclude_file_patterns:
            if pattern.startswith('*') and file_name.endswith(pattern[1:]):
                return True
            elif pattern == file_name:
                return True
        
        # Exclude hidden files (starting with .)
        if file_name.startswith('.'):
            return True
            
        return False

    def _should_exclude_directory(self, app_path: Path, dir_path: Path) -> bool:
        """Determine if a directory should be excluded from the bundle."""
        relative_path = dir_path.relative_to(app_path)
        dir_name = dir_path.name
        
        # Exclude directory patterns
        exclude_patterns = [
            '.git', '__pycache__', 'node_modules', '.venv', 'venv', 
            '.fiber', '.pytest_cache', '.coverage', '.mypy_cache',
            'dist', 'build', '.egg-info'
        ]
        
        # Check if directory name matches any exclude pattern
        for pattern in exclude_patterns:
            if dir_name == pattern or dir_name.endswith(pattern):
                return True
        
        # Exclude hidden directories (starting with .)
        if dir_name.startswith('.'):
            return True
            
        return False

    def _upload_bundle(self, app_version_id: str, bundle_path: Path) -> AppOperationResult:
        """Upload the app bundle to the server."""
        try:
            upload_url = f"{self.api_endpoint}{UPLOAD_API_PATH_TEMPLATE.format(app_version_id=app_version_id)}"
            print(f"[DEBUG] Uploading bundle to: {upload_url}")
            print(f"[DEBUG] Bundle file: {bundle_path} (size: {bundle_path.stat().st_size} bytes)")
            
            with open(bundle_path, 'rb') as f:
                files = {'file': (bundle_path.name, f, 'application/zip')}
                print(f"[DEBUG] Starting upload request...")
                response = requests.post(upload_url, headers=self.headers, files=files, timeout=30)
                print(f"[DEBUG] Upload completed with status: {response.status_code}")
            
            if response.status_code not in (200, 201):
                return AppOperationResult(False, f"Upload failed with status {response.status_code}: {response.text}")
            
            return AppOperationResult(True, "Bundle uploaded successfully")
            
        except requests.exceptions.Timeout:
            return AppOperationResult(False, f"Upload timed out after 30 seconds")
        except Exception as e:
            return AppOperationResult(False, f"Error uploading bundle: {e}")

    def _get_local_app_info_path(self, app_dir: Path) -> Path:
        """Get the path to the local app info file for a given app directory and instance."""
        return app_dir / ".fiber" / self.instance_name / LOCAL_APP_INFO_FILENAME

    def _load_local_app_info(self, app_dir: Path) -> Optional[Dict[str, Any]]:
        """Load local app information from the app directory."""
        app_info_path = self._get_local_app_info_path(app_dir)
        
        if not app_info_path.exists():
            return None
        
        try:
            with open(app_info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    def _save_local_app_info(self, app_dir: Path, app_info: Dict[str, Any]) -> AppOperationResult:
        """Save local app information to the app directory under instance-specific path."""
        try:
            # Ensure .fiber/{instance} directory exists
            fiber_instance_dir = app_dir / ".fiber" / self.instance_name
            fiber_instance_dir.mkdir(parents=True, exist_ok=True)
            
            app_info_path = self._get_local_app_info_path(app_dir)
            
            # Add instance metadata
            app_info["instance_name"] = self.instance_name
            app_info["last_updated"] = datetime.now().isoformat()
            
            with open(app_info_path, 'w', encoding='utf-8') as f:
                json.dump(app_info, f, indent=2)
            
            return AppOperationResult(True, f"Local app info saved to: {app_info_path}")
        except Exception as e:
            return AppOperationResult(False, f"Error saving local app info: {e}")

    def _check_path_changed(self, app_path: Path) -> Tuple[bool, Optional[str]]:
        """Check if the app directory has changed from the last known location."""
        try:
            app_info = self._load_local_app_info(app_path)
            if not app_info:
                return False, None
            
            current_path = str(app_path.resolve())
            previous_path = app_info.get("current_app_dir")
            
            if previous_path and previous_path != current_path:
                return True, previous_path
            
            return False, previous_path
        except Exception:
            return False, None
    
    def _process_oauth_authenticators(self, app_path: Path, manifest: Dict[str, Any]) -> AppOperationResult:
        """Process OAuth authenticator configurations from .fiber/{instance_name}/oauth/ directory."""
        result = AppOperationResult(True, "OAuth authenticators processed")
        
        # Use instance-specific OAuth directory structure
        if self.instance_name:
            oauth_dir = app_path / ".fiber" / self.instance_name / "oauth"
        else:
            # Fallback to old structure for backwards compatibility
            oauth_dir = app_path / ".fiber" / "oauth"
        
        if not oauth_dir.exists():
            print(f"[DEBUG] No OAuth directory found at {oauth_dir}, skipping OAuth authenticator processing")
            return result
        
        print(f"[DEBUG] Processing OAuth authenticators from {oauth_dir}")
        
        # Find all JSON files in the OAuth config directory
        oauth_files = list(oauth_dir.glob("*.json"))
        if not oauth_files:
            print(f"[DEBUG] No OAuth configuration files found in {oauth_dir}")
            return result
        
        app_id = manifest.get("app_id")
        successful_count = 0
        created_authenticators = []  # Track created authenticator IDs
        
        for oauth_file in oauth_files:
            try:
                print(f"[DEBUG] Processing OAuth config: {oauth_file.name}")
                
                with open(oauth_file, 'r', encoding='utf-8') as f:
                    oauth_config = json.load(f)
                
                # Validate required fields
                required_fields = ['display_name', 'system_name', 'authenticator_type', 'client_id', 'client_secret']
                missing_fields = [field for field in required_fields if not oauth_config.get(field)]
                
                if missing_fields:
                    result.add_warning(f"OAuth config {oauth_file.name} missing required fields: {missing_fields}")
                    continue
                
                # Prepare authenticator data for API
                authenticator_data = {
                    "name": oauth_config["display_name"],
                    "system_name": oauth_config["system_name"],
                    "authenticator_type": oauth_config["authenticator_type"],
                    "client_id": oauth_config["client_id"],
                    "client_secret": oauth_config["client_secret"],
                    "scopes": oauth_config.get("scopes", []),
                    "authorize_url": oauth_config.get("authorize_url", ""),
                    "token_url": oauth_config.get("token_url", ""),
                    "redirect_uri": oauth_config.get("redirect_uri", ""),
                    "app_id": app_id
                }
                
                # Send to OAuth authenticators API
                response = requests.post(
                    f"{self.api_endpoint}/api/v1/oauth/authenticators",
                    headers=self.headers,
                    json=authenticator_data,
                    timeout=30
                )
                
                if response.status_code in (200, 201):
                    response_data = response.json()
                    authenticator_id = response_data.get("authenticator_id") or response_data.get("id")
                    
                    # Track the created authenticator
                    created_authenticators.append({
                        "file": oauth_file.name,
                        "display_name": oauth_config["display_name"],
                        "system_name": oauth_config["system_name"],
                        "authenticator_type": oauth_config["authenticator_type"],
                        "authenticator_id": authenticator_id,
                        "created_at": datetime.now().isoformat()
                    })
                    
                    result.add_info(f"OAuth authenticator '{oauth_config['display_name']}' registered successfully")
                    successful_count += 1
                    print(f"[DEBUG] Successfully registered OAuth authenticator: {oauth_config['display_name']}")
                else:
                    error_detail = "Unknown error"
                    try:
                        error_data = response.json()
                        error_detail = error_data.get("detail", f"HTTP {response.status_code}")
                    except:
                        error_detail = f"HTTP {response.status_code}: {response.text}"
                    
                    result.add_warning(f"Failed to register OAuth authenticator '{oauth_config['display_name']}': {error_detail}")
                    print(f"[DEBUG] Failed to register OAuth authenticator {oauth_config['display_name']}: {error_detail}")
                
            except json.JSONDecodeError as e:
                result.add_warning(f"Invalid JSON in OAuth config {oauth_file.name}: {e}")
            except Exception as e:
                result.add_warning(f"Error processing OAuth config {oauth_file.name}: {e}")
        
        if successful_count > 0:
            result.add_info(f"Successfully registered {successful_count} OAuth authenticator(s)")
            
            # Save tracking information to .fiber/{instance}/oauth_tracking.json
            if created_authenticators and self.instance_name:
                tracking_dir = app_path / ".fiber" / self.instance_name
                tracking_dir.mkdir(parents=True, exist_ok=True)
                tracking_file = tracking_dir / "oauth_tracking.json"
                
                try:
                    # Load existing tracking data if it exists
                    existing_data = []
                    if tracking_file.exists():
                        with open(tracking_file, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                    
                    # Merge with new data (avoid duplicates by system_name)
                    existing_names = {item.get("system_name") for item in existing_data}
                    for auth in created_authenticators:
                        if auth["system_name"] not in existing_names:
                            existing_data.append(auth)
                    
                    # Save updated tracking data
                    with open(tracking_file, 'w', encoding='utf-8') as f:
                        json.dump(existing_data, f, indent=2)
                    
                    print(f"[DEBUG] OAuth tracking data saved to {tracking_file}")
                    result.add_info(f"OAuth tracking saved to .fiber/{self.instance_name}/oauth_tracking.json")
                    
                except Exception as e:
                    result.add_warning(f"Failed to save OAuth tracking data: {e}")
        
        return result
