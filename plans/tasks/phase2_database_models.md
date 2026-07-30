# Phase 2: Database Models and Initial Migration

## Overview
This phase implements the core data model with four database tables using SQLAlchemy ORM and creates initial database migration.

## Deliverables
- Database models for User, Upload, Detection, and AuditLog tables
- Model exports module
- Alembic migration script
- Foreign key relationships between tables

## Actionable Tasks

### 2.1 Setup Database Directory
- Create `app/models/` directory
- Create `app/models/__init__.py` for model exports

### 2.2 Create Models

#### 2.2.1 User Model (`app/models/user.py`)
- Implement User model with:
  - User identification fields
  - Authentication and authorization fields
  - Timestamps for user lifecycle
  - Relationships to uploads and audit logs
- Define primary key, unique constraints, indexes

#### 2.2.2 Upload Model (`app/models/upload.py`)
- Implement Upload model with:
  - File metadata (name, path, type, size)
  - Upload status tracking
  - User foreign key reference
  - Scan results and processing metadata
- Implement file storage management logic

#### 2.2.3 Detection Model (`app/models/detection.py`)
- Implement Detection model with:
  - Container number detection data
  - OCR accuracy scores
  - Crop coordinates and bounding boxes
  - Foreign key to upload
- Store type-specific detection results

#### 2.2.4 Audit Log Model (`app/models/audit_log.py`)
- Implement AuditLog model with:
  - Change tracking timestamps
  - Changed field names and old/new values
  - User who performed the change
  - Foreign key to relevant record being edited
- Support detailed change history

### 2.3 Create Database Migration
- Initialize Alembic directory if not exists
- Create `migrations/versions/XXXXXXXXXXXX_initial_migration.py`:
  - Create all four tables with proper constraints
  - Define foreign key relationships
  - Add indexes for performance
  - Include appropriate data types and constraints

## Acceptance Criteria
1. All four models are properly defined with SQLAlchemy relationships
2. Models follow database normalization rules
3. Migration script creates all tables successfully
4. Foreign key constraints maintain data integrity
5. Indexes improve query performance
6. Models can be imported and used in application

## Technical Details
- Use SQLAlchemy declarative base
- Implement proper relationship mappings
- Follow DRY principles in model definitions
- Implement cascade delete/update where appropriate
- Set appropriate nullability and constraint rules