#!/usr/bin/env python3
"""
Import Fix Script for Veritas Logos
Automatically converts problematic relative imports to absolute imports.
"""

import os
import re
import ast
from pathlib import Path
from typing import List, Dict, Set, Tuple
import shutil

class ImportFixer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_root = self.project_root / "src"
        self.backup_dir = self.project_root / "backup_before_import_fix"
        self.fixed_files = []
        self.errors = []
        
    def create_backup(self):
        """Create a backup of all source files before making changes."""
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        
        print(f"Creating backup in: {self.backup_dir}")
        shutil.copytree(self.src_root, self.backup_dir / "src")
        print("✅ Backup created successfully")
    
    def convert_relative_import(self, import_line: str, current_file_path: Path) -> str:
        """Convert a single relative import line to absolute import."""
        
        # Pattern to match relative imports
        pattern = r'from\s+(\.+)([.\w]*)\s+import\s+(.+)'
        match = re.match(pattern, import_line.strip())
        
        if not match:
            return import_line
        
        dots, module_part, import_part = match.groups()
        level = len(dots)
        
        # Convert to absolute import
        if level == 1:
            # Single dot: from .module import something
            # This shouldn't cause the "beyond top-level" error, but let's handle it
            if module_part:
                absolute_module = f"src.{module_part}"
            else:
                # from . import something - need to determine current package
                rel_path = current_file_path.parent.relative_to(self.src_root)
                absolute_module = f"src.{str(rel_path).replace('/', '.')}"
        elif level == 2:
            # Double dot: from ..models import something
            if module_part:
                absolute_module = f"src.{module_part}"
            else:
                # Need to go up one level from current directory
                rel_path = current_file_path.parent.parent.relative_to(self.src_root)
                if str(rel_path) == ".":
                    absolute_module = "src"
                else:
                    absolute_module = f"src.{str(rel_path).replace('/', '.')}"
        elif level == 3:
            # Triple dot: from ...models import something
            if module_part:
                absolute_module = f"src.{module_part}"
            else:
                # Need to go up two levels from current directory
                rel_path = current_file_path.parent.parent.parent.relative_to(self.src_root)
                if str(rel_path) == ".":
                    absolute_module = "src"
                else:
                    absolute_module = f"src.{str(rel_path).replace('/', '.')}"
        else:
            # More than 3 dots - this shouldn't happen in our codebase
            absolute_module = f"src.{module_part}" if module_part else "src"
        
        return f"from {absolute_module} import {import_part}"
    
    def fix_file_imports(self, file_path: Path) -> bool:
        """Fix all imports in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            modified = False
            new_lines = []
            
            for line_num, line in enumerate(lines, 1):
                # Check if this line contains a relative import
                if re.match(r'^\s*from\s+\.+', line):
                    old_line = line
                    new_line = self.convert_relative_import(line, file_path)
                    if new_line != old_line:
                        print(f"  Line {line_num}: '{old_line.strip()}' → '{new_line.strip()}'")
                        modified = True
                    new_lines.append(new_line)
                else:
                    new_lines.append(line)
            
            if modified:
                # Write the modified content back to the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(new_lines))
                self.fixed_files.append(str(file_path.relative_to(self.project_root)))
                return True
            
            return False
            
        except Exception as e:
            error_msg = f"Error fixing {file_path}: {e}"
            print(f"❌ {error_msg}")
            self.errors.append(error_msg)
            return False
    
    def fix_all_imports(self):
        """Fix imports in all Python files with problematic relative imports."""
        print("Starting import fixing process...")
        
        # Get list of files with problematic imports from our analysis
        problematic_files = [
            "src/llm/llm_client.py",
            "src/document_ingestion/document_ingestion_service.py",
            "src/document_ingestion/pdf_parser.py",
            "src/document_ingestion/markdown_parser.py",
            "src/document_ingestion/docx_parser.py",
            "src/document_ingestion/base_parser.py",
            "src/document_ingestion/txt_parser.py",
            "src/models/dashboard.py",
            "src/verification/acvf_repository.py",
            "src/verification/annotation_engine.py",
            "src/api/verification_api.py",
            "src/analytics/adversarial_metrics.py",
            "src/api/routes/verification_routes.py",
            "src/api/routes/billing_routes.py",
            "src/api/routes/verification_status_routes.py",
            "src/api/routes/auth_routes.py",
            "src/api/routes/document_routes.py",
            "src/verification/pipeline/cache.py",
            "src/verification/pipeline/aggregators.py",
            "src/verification/pipeline/verification_pipeline.py",
            "src/verification/pipeline/issue_detection_engine.py",
            "src/verification/pipeline/adapters.py",
            "src/verification/config/chain_loader.py",
            "src/verification/passes/base_pass.py",
            "src/verification/workers/verification_worker.py",
            "src/verification/passes/implementations/acvf_escalation_pass.py",
            "src/verification/passes/implementations/citation_verification_pass.py",
            "src/verification/passes/implementations/claim_extraction_pass.py",
            "src/verification/passes/implementations/ml_enhanced_logic.py",
            "src/verification/passes/implementations/logic_analysis_pass.py",
            "src/verification/passes/implementations/bias_scan_pass.py",
            "src/verification/passes/implementations/ml_enhanced_bias.py"
        ]
        
        total_files = len(problematic_files)
        fixed_count = 0
        
        for file_rel_path in problematic_files:
            file_path = self.project_root / file_rel_path
            if file_path.exists():
                print(f"\nFixing: {file_rel_path}")
                if self.fix_file_imports(file_path):
                    fixed_count += 1
                    print(f"✅ Fixed {file_rel_path}")
                else:
                    print(f"ℹ️  No changes needed in {file_rel_path}")
            else:
                print(f"⚠️  File not found: {file_rel_path}")
        
        print(f"\n{'='*60}")
        print(f"IMPORT FIXING COMPLETED")
        print(f"{'='*60}")
        print(f"Total files processed: {total_files}")
        print(f"Files modified: {fixed_count}")
        print(f"Files with errors: {len(self.errors)}")
        
        if self.errors:
            print(f"\nErrors encountered:")
            for error in self.errors:
                print(f"  ❌ {error}")
        
        print(f"\nFixed files:")
        for fixed_file in self.fixed_files:
            print(f"  ✅ {fixed_file}")
    
    def validate_fixes(self):
        """Run a quick validation to check if imports work after fixes."""
        print(f"\n{'='*60}")
        print("VALIDATING FIXES")
        print(f"{'='*60}")
        
        # Test importing a few key modules
        test_imports = [
            "sys.path.insert(0, 'src'); from src.models.acvf import ACVFRole",
            "sys.path.insert(0, 'src'); from src.models.verification import VerificationResult",
            "sys.path.insert(0, 'src'); from src.llm.llm_client import LLMClient",
        ]
        
        for test_import in test_imports:
            try:
                exec(test_import)
                print(f"✅ {test_import.split(';')[1].strip()}")
            except Exception as e:
                print(f"❌ {test_import.split(';')[1].strip()}: {e}")

def main():
    project_root = "/Users/ajdavis/GitHub/veritas_logos"
    fixer = ImportFixer(project_root)
    
    # Create backup before making changes
    fixer.create_backup()
    
    # Fix all imports
    fixer.fix_all_imports()
    
    # Validate the fixes
    fixer.validate_fixes()
    
    print(f"\n🎉 Import fixing process completed!")
    print(f"📁 Backup available at: {fixer.backup_dir}")
    print(f"🔧 Run the validation script to test the fixes: python validate_implementation.py")

if __name__ == "__main__":
    main() 