from flask import Blueprint, render_template, jsonify
from flask_login import login_required

main_bp = Blueprint('main', __name__)


def compute_dashboard_stats():
    """Shared by index() (initial page render) and /dashboard/data (the
    30-second poll dashboard.html's refreshDashboard() makes) so both return
    the exact same numbers computed the exact same way."""
    from app.extensions import db
    from app.models.upload import Upload
    from app.models.detection import Detection
    from sqlalchemy import func, distinct
    from datetime import datetime, timedelta

    total_scans = db.session.query(func.count(Upload.id)).scalar() or 0
    unique_users = db.session.query(func.count(distinct(Upload.user_id))).scalar() or 0
    raw_avg = db.session.query(func.avg(Detection.confidence_score)).scalar() or 0.0
    avg_confidence = float(raw_avg) * 100

    total_detections = db.session.query(func.count(Detection.id)).scalar() or 0
    valid_detections = db.session.query(func.count(Detection.id)).filter(
        Detection.is_valid == True
    ).scalar() or 0
    validation_success_rate = (valid_detections / total_detections * 100) if total_detections else 0.0

    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    daily_scans_raw = db.session.query(
        func.date(Upload.created_at), func.count(Upload.id)
    ).filter(Upload.created_at >= thirty_days_ago).group_by(
        func.date(Upload.created_at)
    ).order_by(func.date(Upload.created_at)).all()
    daily_scans = {str(row[0]): row[1] for row in daily_scans_raw}

    return {
        'total_scans': total_scans,
        'unique_users': unique_users,
        'avg_confidence': avg_confidence,
        'validation_success_rate': validation_success_rate,
        'daily_scans': daily_scans,
    }


@main_bp.route('/')
@main_bp.route('/index')
@login_required
def index():
    return render_template('main/dashboard.html', stats=compute_dashboard_stats())


@main_bp.route('/dashboard/data')
@login_required
def dashboard_data():
    """Polled every 30s by dashboard.html's refreshDashboard(); previously
    404'd since this route never existed. updateChartData() only reads
    data.stats.daily_scans, but the full stats dict is returned for parity
    with the initial render and any future use of the other fields."""
    return jsonify({'stats': compute_dashboard_stats()})


@main_bp.route('/health')
def health():
    return {'status': 'ok', 'service': 'container-ocr'}
