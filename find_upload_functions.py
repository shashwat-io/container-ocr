import os
import re

# Search for upload-related functions in all Python files
for root, dirs, files in os.walk('app'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            with open(filepath, 'r') as f:
                content = f.read()
                
                # Look for upload routes or functions
                if 'upload' in content.lower():
                    matches = re.finditer(r'def (\w+).*?:.*?(?=def|\Z)', content, re.DOTALL)
                    for match in matches:
                        func_name = match.group(1)
                        func_content = match.group(0)
                        if any(word in func_name.lower() for word in ['upload', 'page']):
                            print(f"{filepath}: {func_name}")
                            # Find route decorators
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                if func_name in line and 'def' in line:
                                    # Look backward for route decorator
                                    j = i - 1
                                    while j >= 0 and (lines[j].strip().startswith('@') or lines[j].strip() == ''):
                                        if lines[j].strip().startswith('@'):
                                            print(f"  Route: {lines[j].strip()}")
                                        j -= 1
                            print()
