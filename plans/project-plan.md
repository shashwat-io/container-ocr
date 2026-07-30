# Shipping Container Scanning System Web App Plan

## Step 1: App Factory, Config, .env.example
**Deliverable:** Core app structure and configuration
**Files:**
- `app/__init__.py` - Flask app factory with blueprint registration
- `app/config.py` - Development/production configurations (USE_STUB_MODEL, DB settings)
- `.env.example` - Environment variables template
- `run.py` - Local development launcher
- `wsgi.py` - Gunicorn entry point
- `requirements.txt` - Dependencies list

## Step 2: Database Models and Initial Migration
**Deliverable:** Four database tables with SQLAlchemy models
**Files:**
- `app/models/__init__.py` - Model exports
- `app/models/user.py` - User model (users table)
- `app/models/upload.py` - Upload model (uploads table)
- `app/models/detection.py` - Detection model (detections table)
- `app/models/audit_log.py` - Audit log model (audit_log table)
- `migrations/versions/XXXXXXXXXXXX_initial_migration.py` - Alembic migration

## Step 3: Auth Blueprint
**Deliverable:** Login/logout with Flask-Login
**Files:**
- `app/auth/__init__.py` - Blueprint init
- `app/auth/routes.py` - Login/logout routes
- `app/auth/views.py` - User loader for Flask-Login
- `templates/auth/login.html` - Login form
- `app/auth/helpers.py` - Password verification utilities

## Step 4: Inference Pipeline with Stub Mode
**Deliverable:** Model loading with stub mode support
**Files:**
- `app/inference/__init__.py` - Inference service initialization
- `app/inference/predictor.py` - Main prediction logic (real vs stub)
- `app/inference/model_utils.py` - Model loading utilities
- `static/js/inference.js` - Client-side upload and display logic

## Step 5: Upload Route and Results Page
**Deliverable:** File upload and crop visualization
**Files:**
- `app/main/__init__.py` - Main blueprint init
- `app/main/routes.py` - Upload and results routes
- `templates/main/upload.html` - Upload interface
- `templates/main/results.html` - Results page with crops
- `static/js/validation.js` - ISO 6346 validation (live)

## Step 6: Save Detections, Records Page
**Deliverable:** Database persistence with records display
**Files:**
- `app/main/routes.py` (extended) - Save and records routes
- `templates/main/records.html` - Search, pagination, filter UI
- `static/js/records.js` - Records listing and search functionality

## Step 7: Record Editing with Audit Log
**Deliverable:** Editable records with audit trail
**Files:**
- `app/main/routes.py` (extended) - Edit routes and update logic
- `templates/main/edit.html` - Edit form with change tracking
- `static/js/edit.js` - Client-side change tracking and validation
- Audit log population logic in edit routes

## Step 8: Admin Blueprint
**Deliverable:** User management for admins
**Files:**
- `app/admin/__init__.py` - Admin blueprint init
- `app/admin/routes.py` - User management routes
- `templates/admin/users.html` - User list with actions
- `static/js/admin.js` - Admin UI functionality

## Step 9: Statistics Dashboard
**Deliverable:** Analytics and metrics display
**Files:**
- `app/main/routes.py` (extended) - Dashboard routes
- `templates/main/dashboard.html` - Statistics visualization
- `static/js/dashboard.js` - Dashboard data visualization

## Step 10: CSV Export
**Deliverable:** Filtered CSV streaming export
**Files:**
- `app/main/utils.py` - Export helper functions
- `templates/main/export.html` - Export controls
- `static/js/export.js` - CSV download functionality
- Export filtering respecting user roles

## Step 10: CSV Export
**Deliverable:** Filtered CSV streaming export
**Files:**
- `app/main/utils.py` - Export helper functions
- `templates/main/export.html` - Export controls
- `static/js/export.js` - CSV download functionality
- Export filtering respecting user roles