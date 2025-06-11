"""
ACVF Configuration Loader.

Loads ACVF configuration from YAML files and creates ACVFConfiguration objects.
"""

import yaml
import os
from typing import Dict, Any, List
from pathlib import Path

from src.models.acvf import ACVFConfiguration, ModelAssignment, ACVFRole


class ACVFConfigLoader:
    """Loads ACVF configuration from YAML files."""
    
    def __init__(self, config_dir: str = None):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir is None:
            # Default to the chains directory
            self.config_dir = Path(__file__).parent / "chains"
        else:
            self.config_dir = Path(config_dir)
    
    def load_config(self, config_file: str = "adversarial_chains.yml") -> ACVFConfiguration:
        """
        Load ACVF configuration from YAML file.
        
        Args:
            config_file: Name of the configuration file
            
        Returns:
            ACVFConfiguration object
        """
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return self._create_acvf_config(config_data)
    
    def _create_acvf_config(self, config_data: Dict[str, Any]) -> ACVFConfiguration:
        """Create ACVFConfiguration from parsed YAML data."""
        
        # Extract model configurations
        models = config_data.get("models", {})
        
        challenger_models = self._create_model_assignments(
            models.get("challengers", []), ACVFRole.CHALLENGER
        )
        
        defender_models = self._create_model_assignments(
            models.get("defenders", []), ACVFRole.DEFENDER
        )
        
        judge_models = self._create_model_assignments(
            models.get("judges", []), ACVFRole.JUDGE
        )
        
        # Extract debate configuration
        debate_config = config_data.get("debate_config", {})
        
        # Extract trigger conditions
        trigger_conditions = config_data.get("trigger_conditions", {})
        
        # Extract advanced settings
        advanced_settings = config_data.get("advanced_settings", {})
        
        # Merge all configuration data
        merged_config = {
            **debate_config,
            **trigger_conditions,
            **advanced_settings
        }
        
        return ACVFConfiguration(
            config_id=f"acvf_{config_data.get('name', 'default')}",
            name=config_data.get("name", "ACVF Configuration"),
            description=config_data.get("description"),
            challenger_models=challenger_models,
            defender_models=defender_models,
            judge_models=judge_models,
            max_rounds_per_debate=debate_config.get("max_rounds_per_debate", 3),
            escalation_threshold=debate_config.get("escalation_threshold", 0.5),
            consensus_threshold=debate_config.get("consensus_threshold", 0.7),
            trigger_conditions=trigger_conditions,
            allow_model_self_assignment=advanced_settings.get("allow_model_self_assignment", False),
            require_unanimous_consensus=advanced_settings.get("require_unanimous_consensus", False),
            enable_meta_judging=advanced_settings.get("enable_meta_judging", False),
            metadata={
                "version": config_data.get("version"),
                "outputs": config_data.get("outputs", {}),
                "integration": config_data.get("integration", {}),
                "advanced_settings": advanced_settings
            }
        )
    
    def _create_model_assignments(self, model_configs: List[Dict[str, Any]], 
                                 role: ACVFRole) -> List[ModelAssignment]:
        """Create ModelAssignment objects from configuration data."""
        assignments = []
        
        for config in model_configs:
            assignment = ModelAssignment(
                model_id=config["model"],
                provider=config["provider"],
                role=role,
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 2000),
                system_prompt_override=config.get("system_prompt_override"),
                metadata=config.get("metadata", {})
            )
            assignments.append(assignment)
        
        return assignments
    
    def list_available_configs(self) -> List[str]:
        """List all available ACVF configuration files."""
        yaml_files = []
        for ext in ['.yml', '.yaml']:
            yaml_files.extend(self.config_dir.glob(f'*{ext}'))
        
        return [f.name for f in yaml_files if 'adversarial' in f.name.lower()]
    
    def validate_config(self, config: ACVFConfiguration) -> List[str]:
        """
        Validate ACVF configuration and return list of issues.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []
        
        # Check required models
        if not config.challenger_models:
            issues.append("At least one challenger model is required")
        
        if not config.defender_models:
            issues.append("At least one defender model is required")
        
        if not config.judge_models:
            issues.append("At least one judge model is required")
        
        # Validate model role assignments
        for model in config.challenger_models:
            if model.role != ACVFRole.CHALLENGER:
                issues.append(f"Challenger model {model.model_id} has incorrect role: {model.role}")
        
        for model in config.defender_models:
            if model.role != ACVFRole.DEFENDER:
                issues.append(f"Defender model {model.model_id} has incorrect role: {model.role}")
        
        for model in config.judge_models:
            if model.role != ACVFRole.JUDGE:
                issues.append(f"Judge model {model.model_id} has incorrect role: {model.role}")
        
        # Validate thresholds
        if not (0.0 <= config.escalation_threshold <= 1.0):
            issues.append("Escalation threshold must be between 0.0 and 1.0")
        
        if not (0.0 <= config.consensus_threshold <= 1.0):
            issues.append("Consensus threshold must be between 0.0 and 1.0")
        
        if config.max_rounds_per_debate < 1 or config.max_rounds_per_debate > 10:
            issues.append("Max rounds per debate must be between 1 and 10")
        
        # Check for model conflicts if self-assignment is disabled
        if not config.allow_model_self_assignment:
            all_models = config.challenger_models + config.defender_models + config.judge_models
            model_keys = [f"{m.provider}:{m.model_id}" for m in all_models]
            
            if len(model_keys) != len(set(model_keys)):
                issues.append("Duplicate models found but self-assignment is disabled")
        
        return issues


def load_default_acvf_config() -> ACVFConfiguration:
    """Load the default ACVF configuration."""
    loader = ACVFConfigLoader()
    return loader.load_config("adversarial_chains.yml")


def load_acvf_config_from_file(config_file: str) -> ACVFConfiguration:
    """Load ACVF configuration from a specific file."""
    loader = ACVFConfigLoader()
    return loader.load_config(config_file) 