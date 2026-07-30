# Test importing models directly without app initialization

import sys
import os

# Directly import SQLAlchemy without Flask
sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')

# Read and test each model file
model_files = [
    'app/models/user.py',
    'app/models/upload.py',
    'app/models/detection.py',
    'app/models/audit_log.py'
]

print("Testing model imports...")
print("=" * 50)

for model_file in model_files:
    print(f"\nTesting {model_file}:")
    try:
        with open(model_file, 'r') as f:
            content = f.read()
        
        # Check if it has proper SQLAlchemy imports
        has_db_import = 'from app import db' in content
        has_relationships = 'relationship(' in content
        has_primary_key = 'primary_key=True' in content
        
        print(f"  ✓ Has db import: {has_db_import}")
        print(f"  ✓ Has relationships: {has_relationships}")
        print(f"  ✓ Has primary key: {has_primary_key}")
        
        # Clean up for eval
        model_globals = {}
        model_globals['db'] = None  # Will be mocked
        
        # Try to eval the model definition
        exec(content, model_globals)
        print(f"  ✓ Can be evaluated")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n" + "=" * 50)

# Check __init__.py
print("\nTesting models/__init__.py:")
try:
    with open('app/models/__init__.py', 'r') as f:
        init_content = f.read()
    
    has_exports = '__all__' in init_content
    print(f"  ✓ Has __all__: {has_exports}")
    
    eval_globals = {}
    exec(init_content, eval_globals)
    print(f"  ✓ Can be evaluated")
    
except Exception as e:
    print(f"  ✗ Error: {e}")

print("\n" + "=" * 50)
print("Model testing complete!")
