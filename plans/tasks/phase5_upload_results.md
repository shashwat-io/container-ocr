# Phase 5: Upload Route and Results Page

## Overview
This phase implements the main upload interface, processing pipeline, and results visualization.

## Deliverables
- Main blueprint for core application routes
- Upload and results routes
- Upload interface template
- Results page with image cropping

## Actionable Tasks

### 5.1 Setup Main Blueprint
- Create `app/main/__init__.py` with Flask blueprint initialization
- Configure the main blueprint for application routes

### 5.2 Implement Core Routes (`app/main/routes.py`)
- Create `/upload` route (GET: form, POST: process upload)
- Create `/results/<upload_id>` route to display detected containers
- Implement file validation (type, size, format)
- Add secure file handling for uploaded images
- Create upload processing pipeline

### 5.3 Upload Interface (`templates/main/upload.html`)
- Design clean, drag-and-drop file upload interface
- Show upload progress and status
- Include file type restrictions and size limits
- Implement responsive design

### 5.4 Results Page (`templates/main/results.html`)
- Display detected container numbers
- Show uploaded image and crop overlays
- Allow user interaction with detected regions
- Implement image zoom and pan functionality
- Display OCR confidence scores

### 5.5 Client-Side Validation (`static/js/validation.js`)
- Implement ISO 6346 container number validation
- Validate container format in real-time
- Display validation error messages
- Support live validation during typing

## Acceptance Criteria
1. Users can upload image files with proper validation
2. Upload process is asynchronous and user-friendly
3. Uploaded images are properly stored and processed
4. Detection results are displayed with high confidence
5. Image and crop overlays are accurately positioned
6. ISO 6346 validation works in real-time
7. User can interact with detected regions

## Technical Details
- Use async upload with progress tracking
- Implement drag-and-drop functionality
- Use Canvas API for image manipulation
- Implement real-time validation
- Add accessibility features