from flask import render_template, request, jsonify, send_file, abort, make_response, current_app
from flask_login import login_required, current_user
from functools import wraps
from werkzeug.exceptions import HTTPException
import logging
import traceback
import os
from datetime import datetime, timedelta

from app.main import main_bp


def _draw_annotated_image(image_bytes, detections, upload_id, app):
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io

        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        draw = ImageDraw.Draw(image)

        for det in detections:
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            label = det.get('label', '')
            conf = det.get('confidence', 0.0)
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            draw.text((x1 + 2, max(0, y1 - 14)), f"{label} {conf:.2f}", fill='red')

        uploads_dir = os.path.join(app.static_folder, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        out_filename = f"{upload_id}_annotated.jpg"
        image.save(os.path.join(uploads_dir, out_filename), 'JPEG')
        return f"/static/uploads/{out_filename}"
    except Exception as e:
        app.logger.error(f"Annotation error: {e}")
        return None


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        from app.models.user import User
        if not current_user.is_authenticated:
            return jsonify({"error": "Authentication required"}), 401
        
        user = User.query.get(current_user.id)
        if not user or not user.is_admin:
            return jsonify({"error": "Admin access required"}), 403
        
        return f(*args, **kwargs)
    return decorated_function


@main_bp.route('/upload', methods=['GET'])
@login_required
def upload_image():
    """Display the image upload interface for OCR processing"""
    try:
        current_app.logger.info(f"Upload page accessed by user: {current_user.id}")
        return render_template('main/upload.html')
    except Exception as e:
        current_app.logger.error(f"Error loading upload page: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": f"Error loading upload page: {str(e)}"}), 500

@main_bp.route('/upload', methods=['POST'])
@login_required
def process_upload():
    """Process one or more uploaded images for OCR/detection."""
    try:
        files = request.files.getlist('file')
        files = [f for f in files if f.filename != '']
        if not files:
            return jsonify({"error": "No file uploaded"}), 400

        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
        max_size = 10 * 1024 * 1024  # 10 MB

        from app.extensions import db
        from app.models.upload import Upload
        from app.models.detection import Detection
        from app.models.audit_log import AuditLog

        all_results = []

        for file in files:
            if not ('.' in file.filename and
                    file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
                all_results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": "Invalid file format. Only PNG, JPG, JPEG, GIF, BMP, TIFF allowed",
                })
                continue

            file.seek(0, 2)
            file_size = file.tell()
            file.seek(0)

            if file_size > max_size:
                all_results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": "File size exceeds 10MB limit",
                })
                continue

            image_bytes = file.read()

            # Verify the bytes really are a decodable image before handing them
            # to the model. The extension check above only looks at the
            # filename, so a .txt renamed to .jpg reached PIL and raised
            # UnidentifiedImageError -- which, in stub mode, escaped to the
            # outer handler and failed the WHOLE batch with a 500 instead of
            # reporting one bad file. verify() consumes the buffer, so this
            # works on a throwaway BytesIO.
            try:
                from PIL import Image as _PILImage
                import io as _io
                _PILImage.open(_io.BytesIO(image_bytes)).verify()
            except Exception as img_err:
                current_app.logger.warning(
                    f"Rejected undecodable upload {file.filename!r}: {img_err}"
                )
                all_results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": "File is not a readable image (corrupt, or not an image at all)",
                })
                continue

            result = current_app.inference_service.predict(image_bytes, file.filename)
            result['filename'] = file.filename

            if result.get('status') == 'success':
                try:
                    upload = Upload(
                        user_id=current_user.id,
                        filename=file.filename,
                        image_size=str(file_size),
                    )
                    db.session.add(upload)
                    db.session.flush()

                    detection_pairs = []
                    for det in result.get('detections', []):
                        detection = Detection(
                            upload_id=upload.id,
                            user_id=current_user.id,
                            confidence_score=det.get('confidence', 0.0),
                            container_code=det.get('container_code', ''),
                            is_valid=det.get('is_valid', False),
                            ocr_raw_output=det.get('ocr_raw_output', ''),
                        )
                        db.session.add(detection)
                        detection_pairs.append((det, detection))

                    db.session.flush()
                    for det, detection in detection_pairs:
                        det['detection_id'] = detection.id

                    db.session.commit()
                    result['upload_id'] = upload.id

                    try:
                        al = AuditLog(
                            user_id=current_user.id,
                            table_name='uploads',
                            record_id=upload.id,
                            change_type='INSERT',
                            field_name='filename',
                            new_value=file.filename,
                            ip_address=request.remote_addr,
                        )
                        db.session.add(al)
                        db.session.commit()
                    except Exception as audit_err:
                        # Was a bare `except Exception: pass`. The upload row is
                        # already committed above, so a failed audit write must
                        # not fail the request -- but it must not vanish either,
                        # and the session has to be rolled back or the next
                        # commit inherits the broken transaction.
                        db.session.rollback()
                        current_app.logger.error(
                            f"Audit log write failed for upload {upload.id}: {audit_err}",
                            exc_info=True,
                        )

                    annotated_url = _draw_annotated_image(
                        image_bytes, result.get('detections', []), upload.id, current_app
                    )
                    if annotated_url:
                        result['annotated_image_url'] = annotated_url

                except Exception as db_err:
                    db.session.rollback()
                    current_app.logger.error(
                        f"DB save error for {file.filename}: {db_err}", exc_info=True
                    )

            all_results.append(result)

        return jsonify({"results": all_results}), 200

    except Exception as e:
        current_app.logger.error(f"Error processing upload: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": f"Error processing upload: {str(e)}"}), 500

@main_bp.route('/records', methods=['GET'])
@login_required
def records_page():
    return render_template('main/records.html')


@main_bp.route('/main/users', methods=['GET'])
@login_required
def get_filter_users():
    """List users for 'Uploaded by' filter dropdowns.

    Admins get every registered user; non-admins only see themselves,
    since they can't see anyone else's uploads anyway.
    """
    from app.models.user import User

    if current_user.is_admin:
        users = User.query.order_by(User.username.asc()).all()
    else:
        users = [User.query.get(current_user.id)]

    return jsonify({'users': [{'id': u.id, 'username': u.username} for u in users if u]})


@main_bp.route('/main/records', methods=['GET'])
@login_required
def get_records_api():
    try:
        from app.extensions import db
        from app.models.upload import Upload
        from app.models.detection import Detection
        from app.models.user import User

        user = User.query.get(current_user.id)
        query = Upload.query if user.is_admin else Upload.query.filter_by(user_id=current_user.id)
        uploads = query.order_by(Upload.created_at.desc()).all()

        static_uploads = os.path.join(current_app.static_folder, 'uploads')
        records = []
        for u in uploads:
            dets = Detection.query.filter_by(upload_id=u.id).all()
            det_count = len(dets)
            max_conf = max((d.confidence_score for d in dets), default=0.0) if dets else 0.0
            codes = [d.container_code for d in dets if d.container_code]
            annotated_url = None
            if os.path.exists(os.path.join(static_uploads, f"{u.id}_annotated.jpg")):
                annotated_url = f"/static/uploads/{u.id}_annotated.jpg"
            records.append({
                'id': u.id,
                'filename': u.filename,
                'image_size': int(u.image_size) if u.image_size and u.image_size.isdigit() else 0,
                'created_at': u.created_at.strftime('%Y-%m-%dT%H:%M:%SZ') if u.created_at else None,
                'detections_count': det_count,
                'max_confidence': max_conf,
                'container_codes': codes,
                'annotated_image_url': annotated_url,
                'user_id': u.user_id,
                'username': u.user.username if u.user else 'Unknown',
            })

        return jsonify({'records': records})
    except Exception as e:
        current_app.logger.error(f"Error fetching records: {e}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@main_bp.route('/main/records/<int:record_id>', methods=['GET'])
@login_required
def get_record_detail_api(record_id):
    try:
        from app.models.upload import Upload
        from app.models.detection import Detection

        upload = Upload.query.get_or_404(record_id)

        # Ownership check. The list endpoint (get_records_api) already scopes
        # rows to the current user, but this detail endpoint did not -- any
        # logged-in user could read any other user's record (filename,
        # container codes, annotated image URL) just by guessing the integer
        # id. Same rule as export_by_ids / download_export.
        if not current_user.is_admin and upload.user_id != current_user.id:
            return jsonify({'error': 'Access denied'}), 403

        detections = Detection.query.filter_by(upload_id=upload.id).all()

        import ast

        det_list = []
        for d in detections:
            coords = {'x1': 0.0, 'y1': 0.0, 'x2': 0.0, 'y2': 0.0}
            if d.ocr_raw_output:
                try:
                    raw = ast.literal_eval(d.ocr_raw_output)
                    coords = {k: float(raw.get(k, 0.0)) for k in ('x1', 'y1', 'x2', 'y2')}
                except Exception:
                    pass
            det_list.append({
                'label': d.container_code or '',
                'container_code': d.container_code or '',
                'confidence': d.confidence_score,
                **coords,
            })

        annotated_url = None
        annotated_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'static', 'uploads', f"{upload.id}_annotated.jpg"
        )
        if os.path.exists(annotated_path):
            annotated_url = f"/static/uploads/{upload.id}_annotated.jpg"

        record = {
            'id': upload.id,
            'filename': upload.filename,
            'image_size': upload.image_size or '0',
            'created_at': upload.created_at.strftime('%Y-%m-%dT%H:%M:%SZ') if upload.created_at else None,
            'detections': det_list,
            'annotated_image_url': annotated_url,
        }
        return jsonify({'record': record})
    except HTTPException:
        # get_or_404 / abort() raise HTTPException, which is an Exception --
        # letting the generic handler below catch it turned a clean 404 (or
        # 403) into a 500 carrying the 404 text in an "error" field.
        raise
    except Exception as e:
        current_app.logger.error(f"Error fetching record {record_id}: {e}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@main_bp.route('/main/records/<int:record_id>/save', methods=['POST'])
@login_required
def save_record_api(record_id):
    return jsonify({'message': 'Record already persisted', 'id': record_id})


@main_bp.route('/detections/<int:detection_id>/code', methods=['POST'])
@login_required
def save_detection_code(detection_id):
    try:
        from app.extensions import db
        from app.models.detection import Detection
        from app.models.audit_log import AuditLog
        from app.inference.ocr import validate_iso6346

        det = Detection.query.get_or_404(detection_id)
        if det.user_id != current_user.id and not current_user.is_admin:
            return jsonify({'error': 'Access denied'}), 403

        data = request.get_json() or {}
        new_code = data.get('container_code', '').strip().upper()

        old_code = det.container_code or ''
        det.container_code = new_code
        det.is_edited = True
        # Re-validate against the manually-entered code rather than leaving
        # is_valid at whatever the original OCR pass set. Previously a
        # hand-typed correction kept the OCR's old is_valid verbatim, so
        # overwriting a valid read with garbage still showed "Valid", and
        # fixing an invalid read by hand still showed "Unverified" -- both
        # records and the CSV export read is_valid as ground truth.
        det.is_valid, _ = validate_iso6346(new_code)

        al = AuditLog(
            user_id=current_user.id,
            table_name='detections',
            record_id=detection_id,
            change_type='UPDATE',
            field_name='container_code',
            old_value=old_code,
            new_value=new_code,
            ip_address=request.remote_addr,
        )
        db.session.add(al)
        db.session.commit()

        return jsonify({'message': 'Saved', 'container_code': new_code, 'is_valid': det.is_valid})
    except HTTPException:
        raise   # keep get_or_404's 404 a 404 -- see get_record_detail_api
    except Exception as e:
        current_app.logger.error(f"save_detection_code error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@main_bp.route('/main/export', methods=['GET'])
@login_required
def export_by_ids():
    ids_param = request.args.get('ids', '')
    try:
        upload_ids = [int(i) for i in ids_param.split(',') if i.strip().isdigit()]
    except ValueError:
        upload_ids = []

    if not upload_ids:
        return jsonify({'error': 'No valid IDs provided'}), 400

    from app.extensions import db
    from app.models.upload import Upload
    from app.models.detection import Detection
    from app.models.user import User
    from app.main.utils import get_original_and_corrected_codes
    import csv, io

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Upload ID', 'Filename', 'Detection #', 'Container Code', 'Original OCR Code', 'User-Corrected Code', 'Confidence', 'Scan Date', 'Image Size', 'Uploaded By'])

    for upload_id in upload_ids:
        upload = Upload.query.get(upload_id)
        if not upload:
            continue
        if not current_user.is_admin and upload.user_id != current_user.id:
            continue
        username = User.query.get(upload.user_id).username if upload.user_id else 'Unknown'
        detections = Detection.query.filter_by(upload_id=upload.id).all()
        codes_by_id = get_original_and_corrected_codes(detections)
        if not detections:
            writer.writerow([upload.id, upload.filename, '', '', '', '', '', upload.created_at.strftime('%Y-%m-%d %H:%M:%S') if upload.created_at else '', upload.image_size or '', username])
        else:
            for i, det in enumerate(detections, start=1):
                original_code, corrected_code = codes_by_id[det.id]
                writer.writerow([
                    upload.id,
                    upload.filename,
                    i,
                    det.container_code or '',
                    original_code,
                    corrected_code,
                    f"{det.confidence_score:.2%}" if det.confidence_score is not None else '',
                    upload.created_at.strftime('%Y-%m-%d %H:%M:%S') if upload.created_at else '',
                    upload.image_size or '',
                    username,
                ])

    response = make_response(output.getvalue())
    response.headers['Content-Type'] = 'text/csv'
    filename = f"export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    response.headers['Content-Disposition'] = f'attachment; filename={filename}'
    return response


@main_bp.route('/export/download/<int:upload_id>', methods=['GET'])
@login_required
def download_export(upload_id):
    from app.extensions import db
    from app.models.upload import Upload
    from app.models.detection import Detection
    from app.models.user import User
    from app.main.utils import get_original_and_corrected_codes
    import csv, io

    upload = Upload.query.get_or_404(upload_id)
    if not current_user.is_admin and upload.user_id != current_user.id:
        abort(403)

    username = User.query.get(upload.user_id).username if upload.user_id else 'Unknown'
    detections = Detection.query.filter_by(upload_id=upload.id).all()
    codes_by_id = get_original_and_corrected_codes(detections)
    scan_date = upload.created_at.strftime('%Y-%m-%d %H:%M:%S') if upload.created_at else ''

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Upload ID', 'Filename', 'Detection #', 'Container Code',
                     'Original OCR Code', 'User-Corrected Code',
                     'Confidence', 'Scan Date', 'Image Size', 'Uploaded By'])
    if not detections:
        writer.writerow([upload.id, upload.filename, '', '', '', '', '', scan_date, upload.image_size or '', username])
    else:
        for i, det in enumerate(detections, start=1):
            original_code, corrected_code = codes_by_id[det.id]
            writer.writerow([
                upload.id,
                upload.filename,
                i,
                det.container_code or '',
                original_code,
                corrected_code,
                f"{det.confidence_score:.2%}" if det.confidence_score is not None else '',
                scan_date,
                upload.image_size or '',
                username,
            ])

    response = make_response(output.getvalue())
    response.headers['Content-Type'] = 'text/csv'
    filename = f"export_{upload_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    response.headers['Content-Disposition'] = f'attachment; filename={filename}'
    return response


@main_bp.route('/export', methods=['GET'])
@login_required
def show_export_page():
    return render_template('main/export.html')


@main_bp.route('/export', methods=['POST'])
@login_required
def generate_export():
    try:
        filters = {}
        
        if request.is_json:
            json_data = request.get_json()
            key_map = {
                'startDate': 'start_date',
                'endDate': 'end_date',
                'minConfidence': 'min_confidence',
                'user': 'user_id',
            }
            for key, value in json_data.items():
                if value and value != 'all':
                    filters[key_map.get(key, key)] = value
        else:
            # Build a filtered copy rather than deleting from the dict while
            # iterating it -- the old form raised
            # "RuntimeError: dictionary changed size during iteration" the
            # moment any field was blank or set to 'all', which is the normal
            # case for a partially-filled export form.
            filters = {
                key: value
                for key, value in request.form.to_dict().items()
                if value and value != 'all'
            }

        from app.models.user import User
        user = User.query.get(current_user.id)
        
        from app.main.utils import generate_csv_export
        response = generate_csv_export(filters, user)

        return response

    except ValueError as e:
        # Malformed filter values (unparseable date, non-numeric confidence or
        # user id) are user input, not server faults -- 400, not 500.
        current_app.logger.warning(f"Rejected export filters {filters!r}: {e}")
        return jsonify({"error": f"Invalid export filter: {e}"}), 400
    except Exception as e:
        # NOTE: this used to read `from app import current_app`, which raises
        # ImportError -- the `app` package exports no such name. Every error
        # path through this handler therefore died with an unhandled
        # ImportError instead of logging and returning JSON. current_app is
        # already imported from flask at module top.
        current_app.logger.error(f"Error generating export: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": f"Error generating export: {str(e)}"}), 500


@main_bp.route('/export/history')
@login_required
def get_export_history():
    from app.extensions import db
    from app.models.user import User
    
    try:
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 20, type=int)
        offset = (page - 1) * limit
        
        user = User.query.get(current_user.id)
        
        # Build query based on user role
        if user.is_admin:
            from app.models.upload import Upload
            from app.models.detection import Detection
            exports = Upload.query \
                .order_by(Upload.created_at.desc()) \
                .offset(offset) \
                .limit(limit) \
                .all()
            
            exports_data = []
            for upload in exports:
                from app.models.detection import Detection
                detection_count = Detection.query.filter_by(upload_id=upload.id).count()
                
                export_data = {
                    "id": upload.id,
                    # Real stored byte size of the uploaded image. This was
                    # `len(upload.filename) + detection_count * 100` -- a
                    # fabricated number with no relationship to anything,
                    # rendered to the operator through formatFileSize() as if
                    # it were a genuine file size.
                    "size": int(upload.image_size) if (upload.image_size or '').isdigit() else 0,
                    "filename": upload.filename,
                    "created_at": upload.created_at.strftime('%Y-%m-%dT%H:%M:%SZ') if upload.created_at else None,
                    "record_count": detection_count,
                    "status": 'completed',
                    "user_id": upload.user_id
                }
                exports_data.append(export_data)

            total = Upload.query.count()
        else:
            from app.models.upload import Upload
            from app.models.detection import Detection
            exports = Upload.query.filter_by(user_id=user.id) \
                .order_by(Upload.created_at.desc()) \
                .offset(offset) \
                .limit(limit) \
                .all()
            
            exports_data = []
            for upload in exports:
                from app.models.detection import Detection
                detection_count = Detection.query.filter_by(upload_id=upload.id).count()
                
                export_data = {
                    "id": upload.id,
                    # See the note in the admin branch above -- this was a
                    # fabricated size, not a real one.
                    "size": int(upload.image_size) if (upload.image_size or '').isdigit() else 0,
                    "filename": upload.filename,
                    "created_at": upload.created_at.strftime('%Y-%m-%dT%H:%M:%SZ') if upload.created_at else None,
                    "record_count": detection_count,
                    "status": 'completed',
                    "user_id": upload.user_id
                }
                exports_data.append(export_data)

            total = Upload.query.filter_by(user_id=user.id).count()
        
        has_more = offset + limit < total
        
        return jsonify({
            "exports": exports_data,
            "page": page,
            "limit": limit,
            "total": total,
            "has_more": has_more
        })
        
    except Exception as e:
        # `from app import current_app` here raised ImportError -- see the
        # note in generate_export.
        current_app.logger.error(f"Error loading export history: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": f"Error loading export history: {str(e)}"}), 500


@main_bp.route('/export/<int:export_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_export(export_id):
    # @admin_required (above) already rejects any non-admin caller with a 403
    # before this function body runs -- there is no non-admin path here to
    # guard against. The dead "owner can also delete their own" check that
    # used to sit here never actually ran for a non-admin, since
    # @admin_required had already returned first; removed rather than left in
    # as a misleading comment describing behaviour that couldn't happen.
    from app.extensions import db

    try:
        from app.models.upload import Upload
        export = Upload.query.get_or_404(export_id)

        db.session.delete(export)
        db.session.commit()
        
        return jsonify({"message": "Export deleted successfully"})

    except HTTPException:
        raise   # keep get_or_404's 404 a 404 -- see get_record_detail_api
    except Exception as e:
        # `from app import current_app` here raised ImportError -- see the
        # note in generate_export.
        db.session.rollback()
        current_app.logger.error(f"Error deleting export: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Error deleting export"}), 500
