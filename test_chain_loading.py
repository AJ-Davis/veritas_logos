#!/usr/bin/env python3
"""
Test script to verify chain configuration loading
"""

import sys
import os
from pathlib import Path

# Add src to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_chain_loading():
    """Test loading verification chain configurations."""
    print("🔧 Testing chain configuration loading...")
    
    try:
        # Import verification components
        from src.verification.config.chain_loader import ChainConfigLoader, create_default_chain_configs
        from src.models.verification import VerificationChainConfig
        
        print("✅ Successfully imported verification modules")
        
        # Test default chain configs
        print("\n🏗️  Testing default chain configs...")
        default_configs = create_default_chain_configs()
        print(f"✅ Created {len(default_configs)} default configurations")
        
        for chain_id, config_data in default_configs.items():
            print(f"  - {chain_id}: {config_data['name']}")
        
        # Test loading from files
        print("\n📂 Testing chain config loader...")
        config_loader = ChainConfigLoader()
        
        # Check if config directory exists
        config_dir = project_root / "src" / "verification" / "config" / "chains"
        if config_dir.exists():
            print(f"✅ Configuration directory found: {config_dir}")
            
            # List available config files
            config_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))
            print(f"📄 Found {len(config_files)} configuration files:")
            for config_file in config_files:
                print(f"  - {config_file.name}")
            
            # Try to load all chains
            print("\n🔄 Loading all chains...")
            try:
                chains = config_loader.load_all_chains()
                print(f"✅ Successfully loaded {len(chains)} chains:")
                
                for chain_id, chain_config in chains.items():
                    print(f"  - {chain_id}: {chain_config.name}")
                    print(f"    Passes: {len(chain_config.passes)}")
                    for pass_config in chain_config.passes:
                        print(f"      • {pass_config.name} ({pass_config.pass_type})")
                
                # Test getting a specific chain
                if 'comprehensive' in chains:
                    print("\n🎯 Testing comprehensive chain:")
                    comprehensive = chains['comprehensive']
                    print(f"  Chain ID: {comprehensive.chain_id}")
                    print(f"  Name: {comprehensive.name}")
                    print(f"  Description: {comprehensive.description}")
                    print(f"  Passes: {len(comprehensive.passes)}")
                    print(f"  Global timeout: {comprehensive.global_timeout_seconds}s")
                    print(f"  Parallel execution: {comprehensive.parallel_execution}")
                    
                    # Test execution order
                    execution_order = comprehensive.get_execution_order()
                    print(f"\n📋 Execution order:")
                    for i, pass_config in enumerate(execution_order, 1):
                        deps = ', '.join([dep.value for dep in pass_config.depends_on]) if pass_config.depends_on else "None"
                        print(f"  {i}. {pass_config.name} (depends on: {deps})")
                
            except Exception as e:
                print(f"❌ Error loading chains: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                
        else:
            print(f"❌ Configuration directory not found: {config_dir}")
        
        print("\n✅ Chain loading test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during chain loading test: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting verification chain loading test\n")
    success = test_chain_loading()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1) 