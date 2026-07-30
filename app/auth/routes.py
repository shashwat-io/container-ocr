from urllib.parse import urlparse

from flask import render_template, request, redirect, url_for, flash, current_app
from flask_login import login_user, logout_user, current_user
from app.auth.views import load_user
from app.auth import auth_bp
from app.admin.helpers import validate_user_input
from app.extensions import db
from app.models import User

from .helpers import verify_user_credentials


def _safe_next_page(next_page):
    """Return `next_page` only if it is a same-site relative path.

    ?next= is attacker-controllable (it is placed in the URL by
    admin.require_admin's redirect, and can be typed by anyone), and it was
    previously passed straight to redirect(). Confirmed: a login POST to
    /auth/login?next=https://evil.example.com/phish returned
    'Location: https://evil.example.com/phish' -- a textbook open redirect,
    which is exactly the primitive a credential-phishing page wants. Anything
    carrying a scheme or a host, and anything not starting with a single '/',
    is discarded in favour of the dashboard.
    """
    if not next_page:
        return url_for('main.index')
    parsed = urlparse(next_page)
    if parsed.scheme or parsed.netloc or not next_page.startswith('/') or next_page.startswith('//'):
        return url_for('main.index')
    return next_page


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')

        user = verify_user_credentials(username, password)

        if user:
            if not user.is_approved:
                flash('Your account is pending admin approval.', 'error')
            elif user.is_active:
                # The checkbox was rendered but never read -- login_user()
                # defaulted to remember=False regardless of what the user
                # chose, so the session cookie always expired at browser
                # close either way.
                remember = request.form.get('remember') == 'on'
                login_user(user, remember=remember)
                try:
                    from app.extensions import db
                    from app.models.audit_log import AuditLog
                    al = AuditLog(
                        user_id=user.id,
                        table_name='users',
                        record_id=user.id,
                        change_type='LOGIN',
                        field_name='session',
                        new_value='logged_in',
                        ip_address=request.remote_addr,
                    )
                    db.session.add(al)
                    db.session.commit()
                except Exception as audit_err:
                    # Was a bare `except Exception: pass`. A failed login-audit
                    # write must not block the login, but silently discarding it
                    # means the audit trail can have holes nobody ever hears
                    # about. Roll back so the poisoned transaction doesn't
                    # follow the request.
                    db.session.rollback()
                    current_app.logger.error(
                        f"Login audit write failed for user {user.id}: {audit_err}",
                        exc_info=True,
                    )
                next_page = _safe_next_page(request.args.get('next'))
                return redirect(next_page)
            else:
                flash('Your account has been deactivated.', 'error')
        else:
            flash('Invalid username or password', 'error')

    return render_template('auth/login.html')


@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        errors = validate_user_input(username, email, password)

        if not password:
            errors.append("Password is required")
        elif password != confirm_password:
            errors.append("Passwords do not match")

        if username and User.query.filter_by(username=username).first():
            errors.append("Username already taken")
        if email and User.query.filter_by(email=email).first():
            errors.append("Email already in use")

        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('auth/signup.html', username=username, email=email)

        new_user = User(username=username, email=email, is_active=True, is_admin=False, is_approved=False)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash('Account created. An administrator must approve your account before you can sign in.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('auth/signup.html')


@auth_bp.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('auth.login'))
