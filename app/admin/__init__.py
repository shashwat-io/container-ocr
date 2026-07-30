from flask import Blueprint
from flask_login import login_required, current_user

from app.models import User

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')


def admin_required(f):
    from functools import wraps

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return {'error': 'Authentication required'}, 401

        if not current_user.is_admin:
            return {'error': 'Admin access required'}, 403

        return f(*args, **kwargs)

    return decorated_function