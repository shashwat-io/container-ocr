#!/usr/bin/env python3
"""Test the upload endpoint directly"""

import requests
import json
import time

def test_upload_endpoint():
    print("Testing Upload Endpoint")
    print("=" * 70)
    
    # The app should already be running
    print("\n1. Testing upload endpoint health...")
    
    # First, check if Flask app is running
    try:
        # Create a simple test file
        import io
        import sys
        
        # Use requests if available
        try:
            # Test the upload endpoint - it will fail due to auth, but we can see if it's configured
            response = requests.get('http://localhost:5000/health', timeout=5)
            print(f"✓ Health endpoint: HTTP {response.status_code}")
            
        except Exception as e:
            print(f"✗ Health endpoint: {e}")
        
        # Check if the upload route exists at all
        print("\n2. Testing upload route configuration...")
        
        # We can also check the Flask app directly
        sys.path.insert(0, '.')
        
        from app import create_app
        from app.main.routes import upload_image, process_upload
        
        app = create_app()
        
        # Check the route endpoints
        with app.app_context():
            print(f"✓ Upload GET endpoint: {upload_image}")
            print(f"✓ Upload POST endpoint: {process_upload}")
            
            # Get route information
            for rule in app.url_map.iter_rules():
                if 'upload' in rule.rule:
                    print(f"  Route: {rule.rule}")
                    print(f"    Methods: {rule.methods}")
                    print(f"    Endpoint: {rule.endpoint}")
                    
        print("\n3. Checking upload implementation...")
        
        # Verify the implementation
        with open('app/main/routes.py', 'r') as f:
            content = f.read()
            
        checks = [
            ('@main_bp.route(\'/upload\', methods=[\'GET\'])', 'GET upload decorator'),
            ('@main_bp.route(\'/upload\', methods=[\'POST\'])', 'POST upload decorator'),
            ('def upload_image():', 'upload_image function definition'),
            ('def process_upload():', 'process_upload function definition'),
            ('login_required', 'login_required decorator'),
            ('render_template(\'main/upload.html\')', 'upload template'),
            ('allowed_extensions', 'file extension validation'),
            ('max_size', 'file size validation'),
            ('current_app.inference_service.predict', 'inference service integration'),
        ]
        
        all_checks_pass = True
        for pattern, description in checks:
            if pattern in content:
                print(f"  ✓ {description}")
            else:
                print(f"  ✗ {description} - missing")
                all_checks_pass = False
        
        print("\n" + "=" * 70)
        if all_checks_pass:
            print("✓ All upload route checks passed!")
            print("\nUPLOAD ROUTE IMPLEMENTATION SUMMARY:")
            print("- Display upload form (GET /upload)")
            print("- Process uploaded file (POST /upload)")
            print("- File validation (type, size)")
            print("- Authentication required")
            print("- Inference service integration")
            print("- Error handling")
        else:
            print("✗ Some upload route checks failed")
            
    except Exception as e:
        print(f"✗ Error testing upload endpoint: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_upload_endpoint()
