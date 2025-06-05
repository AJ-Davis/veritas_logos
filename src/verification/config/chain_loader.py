"""
YAML configuration loader for verification chains.
"""

import yaml
import os
from typing import Dict, List, Optional
from pathlib import Path

from ...models.verification import (
    VerificationChainConfig,
    VerificationPassConfig,
    VerificationPassType,
    VerificationConfigError
)


class ChainConfigLoader:
    """Loads and validates verification chain configurations from YAML files."""
    
    def __init__(self, config_dir: str = "config/chains"):
        """
        Initialize the chain config loader.
        
        Args:
            config_dir: Directory containing YAML configuration files
        """
        self.config_dir = Path(config_dir)
        self.loaded_chains: Dict[str, VerificationChainConfig] = {}
    
    def load_chain_config(self, config_file: str) -> VerificationChainConfig:
        """
        Load a verification chain configuration from a YAML file.
        
        Args:
            config_file: Path to the YAML configuration file
            
        Returns:
            VerificationChainConfig instance
            
        Raises:
            VerificationConfigError: If configuration is invalid
        """
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            raise VerificationConfigError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Validate and create configuration
            chain_config = self._parse_chain_config(config_data)
            self._validate_chain_config(chain_config)
            
            # Cache the loaded configuration
            self.loaded_chains[chain_config.chain_id] = chain_config
            
            return chain_config
            
        except yaml.YAMLError as e:
            raise VerificationConfigError(f"Invalid YAML in {config_file}: {str(e)}")
        except Exception as e:
            raise VerificationConfigError(f"Error loading config {config_file}: {str(e)}")
    
    def load_all_chains(self) -> Dict[str, VerificationChainConfig]:
        """
        Load all verification chain configurations from the config directory.
        
        Returns:
            Dictionary mapping chain IDs to configurations
        """
        if not self.config_dir.exists():
            raise VerificationConfigError(f"Configuration directory not found: {self.config_dir}")
        
        chains = {}
        
        for config_file in self.config_dir.glob("*.yml"):
            try:
                chain_config = self.load_chain_config(config_file.name)
                chains[chain_config.chain_id] = chain_config
            except VerificationConfigError as e:
                print(f"Warning: Failed to load {config_file}: {e}")
        
        for config_file in self.config_dir.glob("*.yaml"):
            try:
                chain_config = self.load_chain_config(config_file.name)
                chains[chain_config.chain_id] = chain_config
            except VerificationConfigError as e:
                print(f"Warning: Failed to load {config_file}: {e}")
        
        self.loaded_chains.update(chains)
        return chains
    
    def get_chain_config(self, chain_id: str) -> Optional[VerificationChainConfig]:
        """
        Get a loaded chain configuration by ID.
        
        Args:
            chain_id: Chain identifier
            
        Returns:
            VerificationChainConfig or None if not found
        """
        return self.loaded_chains.get(chain_id)
    
    def _parse_chain_config(self, config_data: Dict) -> VerificationChainConfig:
        """
        Parse YAML configuration data into a VerificationChainConfig object.
        
        Args:
            config_data: Parsed YAML data
            
        Returns:
            VerificationChainConfig instance
        """
        if not isinstance(config_data, dict):
            raise VerificationConfigError("Configuration must be a YAML object")
        
        # Extract chain-level configuration
        chain_id = config_data.get('chain_id')
        if not chain_id:
            raise VerificationConfigError("Missing required field: chain_id")
        
        name = config_data.get('name')
        if not name:
            raise VerificationConfigError("Missing required field: name")
        
        description = config_data.get('description')
        global_timeout = config_data.get('global_timeout_seconds', 3600)
        parallel_execution = config_data.get('parallel_execution', False)
        stop_on_failure = config_data.get('stop_on_failure', True)
        metadata = config_data.get('metadata', {})
        
        # Parse verification passes
        passes_data = config_data.get('passes', [])
        if not passes_data:
            raise VerificationConfigError("Chain must have at least one verification pass")
        
        passes = []
        for pass_data in passes_data:
            pass_config = self._parse_pass_config(pass_data)
            passes.append(pass_config)
        
        return VerificationChainConfig(
            chain_id=chain_id,
            name=name,
            description=description,
            passes=passes,
            global_timeout_seconds=global_timeout,
            parallel_execution=parallel_execution,
            stop_on_failure=stop_on_failure,
            metadata=metadata
        )
    
    def _parse_pass_config(self, pass_data: Dict) -> VerificationPassConfig:
        """
        Parse a verification pass configuration.
        
        Args:
            pass_data: Pass configuration data
            
        Returns:
            VerificationPassConfig instance
        """
        pass_type_str = pass_data.get('type')
        if not pass_type_str:
            raise VerificationConfigError("Pass missing required field: type")
        
        try:
            pass_type = VerificationPassType(pass_type_str)
        except ValueError:
            valid_types = [t.value for t in VerificationPassType]
            raise VerificationConfigError(
                f"Invalid pass type '{pass_type_str}'. Valid types: {valid_types}"
            )
        
        name = pass_data.get('name')
        if not name:
            raise VerificationConfigError("Pass missing required field: name")
        
        return VerificationPassConfig(
            pass_type=pass_type,
            name=name,
            description=pass_data.get('description'),
            enabled=pass_data.get('enabled', True),
            timeout_seconds=pass_data.get('timeout_seconds', 300),
            max_retries=pass_data.get('max_retries', 3),
            retry_delay_seconds=pass_data.get('retry_delay_seconds', 10),
            parameters=pass_data.get('parameters', {}),
            depends_on=pass_data.get('depends_on', [])
        )
    
    def _validate_chain_config(self, chain_config: VerificationChainConfig):
        """
        Validate a chain configuration for logical consistency.
        
        Args:
            chain_config: Configuration to validate
            
        Raises:
            VerificationConfigError: If configuration is invalid
        """
        # Check for duplicate pass names
        pass_names = [p.name for p in chain_config.passes]
        if len(pass_names) != len(set(pass_names)):
            raise VerificationConfigError("Duplicate pass names found in chain")
        
        # Validate dependencies
        for pass_config in chain_config.passes:
            for dependency in pass_config.depends_on:
                if dependency not in pass_names:
                    raise VerificationConfigError(
                        f"Pass '{pass_config.name}' depends on unknown pass '{dependency}'"
                    )
        
        # Check for circular dependencies
        self._check_circular_dependencies(chain_config.passes)
    
    def _check_circular_dependencies(self, passes: List[VerificationPassConfig]):
        """
        Check for circular dependencies in pass configuration.
        
        Args:
            passes: List of pass configurations
            
        Raises:
            VerificationConfigError: If circular dependencies are found
        """
        pass_map = {p.name: p for p in passes}
        
        def has_cycle(pass_name: str, visited: set, rec_stack: set) -> bool:
            visited.add(pass_name)
            rec_stack.add(pass_name)
            
            pass_config = pass_map.get(pass_name)
            if pass_config:
                for dependency in pass_config.depends_on:
                    if dependency not in visited:
                        if has_cycle(dependency, visited, rec_stack):
                            return True
                    elif dependency in rec_stack:
                        return True
            
            rec_stack.remove(pass_name)
            return False
        
        visited = set()
        for pass_config in passes:
            if pass_config.name not in visited:
                if has_cycle(pass_config.name, visited, set()):
                    raise VerificationConfigError("Circular dependency detected in verification chain")


def create_default_chain_configs():
    """Create default verification chain configurations."""
    
    # Standard verification chain
    standard_chain = {
        'chain_id': 'standard_verification',
        'name': 'Standard Document Verification',
        'description': 'Standard verification chain for document quality control',
        'global_timeout_seconds': 3600,
        'parallel_execution': False,
        'stop_on_failure': False,
        'passes': [
            {
                'type': 'claim_extraction',
                'name': 'extract_claims',
                'description': 'Extract claims from document content',
                'timeout_seconds': 300,
                'max_retries': 2,
                'parameters': {
                    'model': 'gpt-4',
                    'temperature': 0.1,
                    'max_claims': 50
                }
            },
            {
                'type': 'citation_check',
                'name': 'verify_citations',
                'description': 'Verify document citations',
                'timeout_seconds': 600,
                'max_retries': 3,
                'depends_on': ['extract_claims'],
                'parameters': {
                    'model': 'gpt-4',
                    'deep_check': True
                }
            },
            {
                'type': 'logic_analysis',
                'name': 'analyze_logic',
                'description': 'Analyze logical consistency',
                'timeout_seconds': 400,
                'max_retries': 2,
                'depends_on': ['extract_claims'],
                'parameters': {
                    'model': 'claude-3-opus',
                    'check_fallacies': True
                }
            },
            {
                'type': 'bias_scan',
                'name': 'scan_bias',
                'description': 'Scan for bias in content',
                'timeout_seconds': 300,
                'max_retries': 2,
                'depends_on': ['extract_claims'],
                'parameters': {
                    'model': 'gpt-4',
                    'bias_types': ['political', 'cultural', 'statistical']
                }
            }
        ]
    }
    
    # Fast verification chain
    fast_chain = {
        'chain_id': 'fast_verification',
        'name': 'Fast Document Verification',
        'description': 'Lightweight verification for quick turnaround',
        'global_timeout_seconds': 1200,
        'parallel_execution': True,
        'stop_on_failure': False,
        'passes': [
            {
                'type': 'claim_extraction',
                'name': 'extract_claims_fast',
                'description': 'Quick claim extraction',
                'timeout_seconds': 120,
                'max_retries': 1,
                'parameters': {
                    'model': 'gpt-3.5-turbo',
                    'temperature': 0.2,
                    'max_claims': 20
                }
            },
            {
                'type': 'citation_check',
                'name': 'basic_citation_check',
                'description': 'Basic citation verification',
                'timeout_seconds': 180,
                'max_retries': 1,
                'depends_on': ['extract_claims_fast'],
                'parameters': {
                    'model': 'gpt-3.5-turbo',
                    'deep_check': False
                }
            }
        ]
    }
    
    return {
        'standard_verification': standard_chain,
        'fast_verification': fast_chain
    }