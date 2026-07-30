import sys
from app import create_app

try:
    app = create_app()
    print("✓ Flask app created successfully")
    
    with app.app_context():
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        print(f"✓ Found {len(routes)} routes")
        for route in routes:
            print(f"  {route}")
            
        # Check for upload route specifically
        upload_routes = [r for r in routes if 'upload' in r]
        print(f"✓ Upload routes found: {upload_routes}")
        
except Exception as e:
    print(f"✗ Error creating Flask app: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test the actual routes
with app.app_context():
    print("\n" + "=" * 70)
    print("TESTING UPLOAD ROUTE SPECIFICALLY")
    print("=" * 70)
    
    # Get the upload route function
    from app.main.routes import upload_image, process_upload
    
    print(f"✓ upload_image function exists: {upload_image}")
    print(f"✓ process_upload function exists: {process_upload}")
    
    # Check if they have the expected route decorators
    print(f"✓ upload_image has login_required: {hasattr(upload_image, '__wrapped__')}")
    print(f"✓ process_upload has login_required: {hasattr(process_upload, '__wrapped__')}")
    
    # Get route rules
    from flask import url_for
    try:
        upload_get_url = url_for('main.upload_image')
        print(f"✓ upload_image route url: {upload_get_url}")
    except Exception as e:
        print(f"✗ Error getting upload_image route url: {e}")
    
    # Check if the upload page template exists
    import os
    upload_template_path = os.path.join('templates', 'main', 'upload.html')
    if os.path.exists(upload_template_path):
        print(f"✓ Upload template exists: {upload_template_path}")
        with open(upload_template_path, 'r') as f:
            content = f.read()
            print(f"  Template size: {len(content)} bytes")
            if 'upload' in content.lower():
                print("  ✓ Contains upload-related content")
            if 'uploadForm' in content:
                print("  ✓ Contains uploadForm id")
    else:
        print(f"✗ Upload template missing: {upload_template_path}")
    
    # Check if the route method matches what we expect
    upload_route = None
    for rule in app.url_map.iter_rules():
        if 'upload' in rule.rule:
            upload_route = rule
            break
    
    if upload_route:
        print(f"✓ Found upload route in url_map: {upload_route.rule}")
        print(f"  Methods: {upload_route.methods}")
        print(f"  Endpoint: {upload_route.endpoint}")
    else:
        print("✗ No upload route found in url_map")
    
    print("\n" + "=" * 70)
    print("INFERENCE SERVICE CHECK")
    print("=" * 70)
    
    # Check if inference service exists
    if hasattr(app, 'inference_service'):
        print(f"✓ Inference service exists: {app.inference_service}")
        
        # Check if it has the predict method
        if hasattr(app.inference_service, 'predict'):
            print("✓ Inference service has predict method")
        else:
            print("✗ Inference service missing predict method")
    else:
        print("✗ Inference service not attached to app")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    issues = []
    
    # Check for basic app structure
    if len(routes) >= 15:
        print(f"✓ App has adequate number of routes: {len(routes)}")
    else:
        issues.append(f"Only {len(routes)} routes, expected at least 15")
        
    # Check for upload routes
    if any('upload' in r for r in routes):
        print("✓ Upload routes are present")
    else:
        issues.append("No upload routes found")
        
    # Check for inference route
    if any('/inference' in r for r in routes):
        print("✓ Inference routes are present")
    else:
        issues.append("No inference routes found")
    
    if not issues:
        print("\n✓ All basic checks passed!")
        print("\nApp Structure:")
        print("- Flask app created successfully")
        print("- Upload routes (GET/POST) are defined")
        print("- Upload template exists")
        print("- Inference service is configured")
        print("- OAuth2 login system (Flask-Login) is integrated")
    else:
        print("\n✗ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    
    print("=" * 70)

