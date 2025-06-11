#!/usr/bin/env python3
"""
Import Analysis Script for Veritas Logos
Analyzes all Python files to identify problematic relative imports and generate a fix plan.
"""

import os
import re
import ast
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict

class ImportAnalyzer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_root = self.project_root / "src"
        self.problematic_imports = []
        self.all_imports = []
        self.missing_init_files = []
        
    def analyze_file(self, file_path: Path) -> Dict:
        """Analyze a single Python file for import statements."""
        result = {
            'file': str(file_path.relative_to(self.project_root)),
            'relative_imports': [],
            'problematic_imports': [],
            'all_imports': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST to get import information
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module:
                        import_info = {
                            'line': node.lineno,
                            'module': node.module,
                            'names': [alias.name for alias in (node.names or [])],
                            'level': node.level
                        }
                        result['all_imports'].append(import_info)
                        
                        # Check for relative imports
                        if node.level > 0:
                            result['relative_imports'].append(import_info)
                            
                            # Check for problematic relative imports (going beyond top-level)
                            if node.level > 1 or (node.level == 1 and '../' in str(node.module or '')):
                                result['problematic_imports'].append(import_info)
                                
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            
        return result
    
    def find_missing_init_files(self):
        """Find directories that should have __init__.py files."""
        for root, dirs, files in os.walk(self.src_root):
            root_path = Path(root)
            
            # Skip __pycache__ directories
            if '__pycache__' in root_path.parts:
                continue
                
            # Check if this directory contains Python files
            has_python_files = any(f.endswith('.py') for f in files)
            has_init = '__init__.py' in files
            
            if has_python_files and not has_init:
                self.missing_init_files.append(str(root_path.relative_to(self.project_root)))
    
    def scan_project(self):
        """Scan the entire project for import issues."""
        print("Scanning project for import issues...")
        
        # Find all Python files
        python_files = list(self.src_root.rglob("*.py"))
        
        for file_path in python_files:
            if '__pycache__' not in str(file_path):
                analysis = self.analyze_file(file_path)
                if analysis['problematic_imports']:
                    self.problematic_imports.append(analysis)
                self.all_imports.append(analysis)
        
        # Find missing __init__.py files
        self.find_missing_init_files()
    
    def generate_report(self):
        """Generate a comprehensive report of import issues."""
        print("\n" + "="*60)
        print("IMPORT ANALYSIS REPORT")
        print("="*60)
        
        print(f"\nProject Root: {self.project_root}")
        print(f"Source Root: {self.src_root}")
        
        print(f"\nFiles Scanned: {len(self.all_imports)}")
        print(f"Files with Problematic Imports: {len(self.problematic_imports)}")
        print(f"Missing __init__.py files: {len(self.missing_init_files)}")
        
        # Report missing __init__.py files
        if self.missing_init_files:
            print("\n" + "-"*40)
            print("MISSING __init__.py FILES:")
            print("-"*40)
            for missing in self.missing_init_files:
                print(f"  {missing}")
        
        # Report problematic imports
        if self.problematic_imports:
            print("\n" + "-"*40)
            print("PROBLEMATIC IMPORTS:")
            print("-"*40)
            
            for file_analysis in self.problematic_imports:
                print(f"\nFile: {file_analysis['file']}")
                for imp in file_analysis['problematic_imports']:
                    print(f"  Line {imp['line']}: from {'.' * imp['level']}{imp['module']} import {', '.join(imp['names'])}")
        
        # Generate fix suggestions
        print("\n" + "-"*40)
        print("SUGGESTED FIXES:")
        print("-"*40)
        
        print("\n1. Create missing __init__.py files:")
        for missing in self.missing_init_files:
            print(f"   touch {missing}/__init__.py")
        
        print("\n2. Convert relative imports to absolute imports:")
        print("   Replace patterns like:")
        print("   - 'from ..models.acvf import ACVFRole' → 'from src.models.acvf import ACVFRole'")
        print("   - 'from ...models.verification import' → 'from src.models.verification import'")
        
        print("\n3. Update validation script to handle package structure:")
        print("   Ensure sys.path is configured correctly for 'src' as the package root")
        
        return {
            'total_files': len(self.all_imports),
            'problematic_files': len(self.problematic_imports),
            'missing_init_files': self.missing_init_files,
            'problematic_imports': self.problematic_imports
        }

def main():
    project_root = "/Users/ajdavis/GitHub/veritas_logos"
    analyzer = ImportAnalyzer(project_root)
    analyzer.scan_project()
    report = analyzer.generate_report()
    
    # Save detailed report to file
    with open("import_analysis_report.txt", "w") as f:
        f.write("DETAILED IMPORT ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write("PROBLEMATIC IMPORTS BY FILE:\n")
        f.write("-"*30 + "\n")
        for file_analysis in analyzer.problematic_imports:
            f.write(f"\n{file_analysis['file']}:\n")
            for imp in file_analysis['problematic_imports']:
                f.write(f"  Line {imp['line']}: from {'.' * imp['level']}{imp['module']} import {', '.join(imp['names'])}\n")
        
        f.write(f"\nMISSING __init__.py FILES:\n")
        f.write("-"*25 + "\n")
        for missing in analyzer.missing_init_files:
            f.write(f"{missing}\n")
    
    print(f"\nDetailed report saved to: import_analysis_report.txt")
    print(f"Summary: {report['problematic_files']} files need import fixes")

if __name__ == "__main__":
    main() 