from datetime import datetime, timedelta
from flask import make_response
import io
import csv
from sqlalchemy import extract, and_
from flask_login import current_user


def get_original_and_corrected_codes(detections):
    """
    Given an iterable of Detection objects, return {detection.id: (original_code, corrected_code)}.

    original_code is the OCR value as first detected (container_code if never
    edited, otherwise the old_value of the earliest audit_log row for that
    detection's container_code field). corrected_code is the current live
    container_code if the detection was edited, otherwise ''.

    Does a single batched audit_log query for all edited detections rather
    than one query per detection.
    """
    from app.models.audit_log import AuditLog

    detections = list(detections)
    edited_ids = [d.id for d in detections if d.is_edited]

    earliest_old_value = {}
    if edited_ids:
        rows = (
            AuditLog.query
            .filter(
                AuditLog.table_name == 'detections',
                AuditLog.field_name == 'container_code',
                AuditLog.record_id.in_(edited_ids),
            )
            .order_by(AuditLog.record_id, AuditLog.changed_at.asc())
            .all()
        )
        for row in rows:
            earliest_old_value.setdefault(row.record_id, row.old_value)

    result = {}
    for d in detections:
        if d.is_edited:
            original_code = earliest_old_value.get(d.id) or ''
            corrected_code = d.container_code or ''
        else:
            original_code = d.container_code or ''
            corrected_code = ''
        result[d.id] = (original_code, corrected_code)
    return result


def generate_csv_export(filters=None, user=None):
    """
    Generate CSV export based on filters and user permissions.
    
    Args:
        filters (dict): Filter criteria (start_date, end_date, min_confidence, etc.)
        user: Current user object for permission checking
    
    Returns:
        tuple: (response, status_code) - CSV response and status
    """
    from app.extensions import db
    from app.models.upload import Upload
    from app.models.detection import Detection
    
    # Build query with filters
    query = Upload.query

    # Detection must be joined AT MOST ONCE. Each filter below used to call
    # .join(Detection) independently, so selecting any two of them at the same
    # time (e.g. min confidence + container code) produced a second join of the
    # same table and SQLite failed the whole request with
    # "OperationalError: ambiguous column name: detections.confidence_score".
    _joined_detection = False

    def _join_detection(q):
        nonlocal _joined_detection
        if not _joined_detection:
            q = q.join(Detection)
            _joined_detection = True
        return q

    # Apply date range filter
    if filters and 'start_date' in filters and filters['start_date']:
        start_date = datetime.strptime(filters['start_date'], '%Y-%m-%d')
        query = query.filter(Upload.created_at >= start_date)

    if filters and 'end_date' in filters and filters['end_date']:
        end_date = datetime.strptime(filters['end_date'], '%Y-%m-%d') + timedelta(days=1)
        query = query.filter(Upload.created_at <= end_date)

    # Apply minimum confidence filter
    if filters and 'min_confidence' in filters:
        # The export form sends a PERCENTAGE (0-100, "Minimum Confidence (%)",
        # step=5) but Detection.confidence_score is stored as a 0-1 fraction.
        # The raw string was previously compared straight against the float
        # column, so SQLite's type-affinity rules made every comparison false
        # and the filter silently returned an empty CSV -- confirmed: a filter
        # of 50 returned 0 data rows where 0.5 returned 11. records.js already
        # divides by 100 for its own client-side version of this filter.
        min_confidence = float(filters['min_confidence']) / 100.0
        query = _join_detection(query).filter(Detection.confidence_score >= min_confidence)

    # Apply user filter
    if filters and 'user_id' in filters:
        query = query.filter(Upload.user_id == int(filters['user_id']))

    # Apply container code filter
    if filters and 'container_code' in filters:
        if filters['container_code'].lower() != 'all':
            query = _join_detection(query).filter(Detection.container_code.ilike(f"%{filters['container_code']}%"))

    # Apply status filter
    if filters and 'status' in filters and filters['status'] != 'all':
        if filters['status'] == 'valid':
            query = _join_detection(query).filter(Detection.is_valid == True)
        elif filters['status'] == 'invalid':
            query = _join_detection(query).filter(Detection.is_valid == False)
        elif filters['status'] == 'flagged':
            query = _join_detection(query).filter(Detection.is_flagged == True)

    # Access control
    from app.models.user import User
    if user and not user.is_admin:
        query = query.filter(Upload.user_id == user.id)
    
    # Order by creation date
    query = query.order_by(Upload.created_at.desc())
    
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        'Upload ID', 'Filename', 'Detection #', 'Container Code',
        'Original OCR Code', 'User-Corrected Code',
        'Confidence', 'Scan Date', 'Image Size', 'Uploaded By'
    ])

    for upload in query.all():
        from app.models.user import User
        username = User.query.get(upload.user_id).username if upload.user_id else 'Unknown'
        detections = Detection.query.filter_by(upload_id=upload.id).all()
        codes_by_id = get_original_and_corrected_codes(detections)
        scan_date = upload.created_at.strftime('%Y-%m-%d %H:%M:%S') if upload.created_at else ''

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
    filename = f"exports_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    response.headers['Content-Disposition'] = f'attachment; filename={filename}'
    return response
