from flask import Blueprint, jsonify, Flask
from flask_wtf import CSRFProtect
from app.config import get_config
from app.inference import InferenceService
from app.extensions import db, login_manager, bcrypt, migrate
from app.auth.views import load_user
from datetime import timezone, timedelta
import os

login_manager.user_loader(load_user)

_IST = timezone(timedelta(hours=5, minutes=30))

def _to_ist_filter(dt, fmt=None):
    """Jinja2 filter: convert a UTC-naive datetime to IST and format with 'IST' label."""
    if dt is None:
        return ''
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ist_dt = dt.astimezone(_IST)
    base = ist_dt.strftime(fmt) if fmt else ist_dt.strftime('%Y-%m-%d %H:%M:%S')
    return base + ' IST'


def create_app():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    app = Flask(
        __name__,
        template_folder=os.path.join(base_dir, 'templates'),
        static_folder=os.path.join(base_dir, 'static')
    )
    config_name = os.environ.get('FLASK_CONFIG', 'prod')
    app.config.from_object(get_config(config_name))

    # ProductionConfig reads SECRET_KEY and DATABASE_URL straight from the
    # environment with no default, so both land as None when unset. Flask then
    # starts happily and only fails later, at the first request that touches a
    # session or the database -- an unset SECRET_KEY surfaces as a runtime
    # error on login, not as a startup failure. Check them here instead.
    if config_name == 'prod':
        missing = [
            name for name, value in (
                ('SECRET_KEY', app.config.get('SECRET_KEY')),
                ('DATABASE_URL', app.config.get('SQLALCHEMY_DATABASE_URI')),
            ) if not value
        ]
        if missing:
            raise RuntimeError(
                f"FLASK_CONFIG=prod requires {' and '.join(missing)} to be set "
                "in the environment (or .env). Refusing to start half-configured."
            )

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    bcrypt.init_app(app)
    migrate.init_app(app, db)
    csrf = CSRFProtect(app)
    from app.main.routes import main_bp
    from app.auth.routes import auth_bp
    from app.admin.routes import admin_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(admin_bp)
    app.jinja_env.filters['to_ist'] = _to_ist_filter
    app.inference_service = InferenceService.init_app(app)
    from app.cli import register_cli
    register_cli(app)
    return app
