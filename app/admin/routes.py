from flask import render_template, request, jsonify, redirect, url_for, flash, current_app
from flask_login import current_user, login_required
import traceback
from werkzeug.security import generate_password_hash
import uuid

from app.admin import admin_bp
from app.admin.helpers import validate_user_input, reset_user_password, can_manage_user
from app.models import User
from app.extensions import db


def require_admin(f):
    from functools import wraps

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('auth.login', next=request.url))

        if not current_user.is_admin:
            flash('Admin access required', 'error')
            return redirect(url_for('main.index'))

        return f(*args, **kwargs)

    return decorated_function


@admin_bp.route('/dashboard')
@require_admin
def dashboard():
    try:
        total_users = User.query.count()
        active_users = User.query.filter_by(is_active=True).count()
        admin_users = User.query.filter_by(is_admin=True).count()

        recent_users = User.query.order_by(User.created_at.desc()).limit(10).all()

        return render_template('admin/dashboard.html',
                               total_users=total_users,
                               active_users=active_users,
                               admin_users=admin_users,
                               recent_users=recent_users)

    except Exception as e:
        current_app.logger.error(f"Admin dashboard error: {str(e)}\n{traceback.format_exc()}")
        flash('Error loading admin dashboard', 'error')
        return redirect(url_for('main.index'))


@admin_bp.route('/users/create', methods=['GET', 'POST'])
@require_admin
def create_user():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '').strip()
        is_admin = request.form.get('is_admin') == 'on'

        from app.admin.helpers import validate_user_input
        errors = validate_user_input(username, email, password)

        if not password:
            errors.append("Password is required")

        if username and User.query.filter_by(username=username).first():
            errors.append("Username already taken")
        if email and User.query.filter_by(email=email).first():
            errors.append("Email already in use")

        if errors:
            for error in errors:
                flash(error, 'error')
            return redirect(url_for('admin.create_user'))

        new_user = User(username=username, email=email, is_active=True, is_admin=is_admin)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.flush()
        from app.models.audit_log import AuditLog
        db.session.add(AuditLog(user_id=current_user.id, table_name='users', record_id=new_user.id,
            change_type='INSERT', field_name='username', new_value=username, ip_address=request.remote_addr))
        db.session.commit()

        flash(f'User "{username}" created successfully', 'success')
        return redirect(url_for('admin.list_users'))

    return render_template('admin/create_user.html')


@admin_bp.route('/users')
@require_admin
def list_users():
    try:
        page = request.args.get('page', 1, type=int)
        per_page = 20
        search = request.args.get('search', '', type=str)

        query = User.query

        if search:
            query = query.filter(User.username.ilike(f'%{search}%') |
                                User.email.ilike(f'%{search}%'))

        users = query.order_by(User.created_at.desc()).paginate(page=page, per_page=per_page)

        return render_template('admin/users.html',
                               users=users,
                               search=search,
                               page=page,
                               can_manage_user=lambda u: can_manage_user(u, current_user))

    except Exception as e:
        current_app.logger.error(f"List users error: {str(e)}\n{traceback.format_exc()}")
        flash('Error loading users', 'error')
        return redirect(url_for('admin.dashboard'))


@admin_bp.route('/users/<int:user_id>/edit', methods=['GET'])
@require_admin
def edit_user_form(user_id):
    try:
        user = User.query.get_or_404(user_id)

        if not can_manage_user(user, current_user):
            flash('You cannot edit this user', 'error')
            return redirect(url_for('admin.list_users'))

        from app.models import Upload
        upload_count = Upload.query.filter_by(user_id=user_id).count()

        return render_template('admin/edit_user.html',
                               user=user,
                               upload_count=upload_count)

    except Exception as e:
        current_app.logger.error(f"Edit user form error: {str(e)}\n{traceback.format_exc()}")
        flash('Error loading user edit form', 'error')
        return redirect(url_for('admin.list_users'))


@admin_bp.route('/users/<int:user_id>/edit', methods=['POST'])
@require_admin
def edit_user(user_id):
    try:
        user = User.query.get_or_404(user_id)

        if not can_manage_user(user, current_user):
            return jsonify({'error': 'You cannot edit this user'}), 403

        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        is_active = request.form.get('is_active') == 'on'
        is_admin = request.form.get('is_admin') == 'on'
        new_password = request.form.get('new_password', '').strip()

        errors = validate_user_input(username, email, new_password if new_password else None)

        if not username:
            errors.append("Username is required")
        elif User.query.filter_by(username=username).first() and User.query.filter_by(username=username).first().id != user.id:
            errors.append("Username already taken")

        if not email:
            errors.append("Email is required")
        elif User.query.filter_by(email=email).first() and User.query.filter_by(email=email).first().id != user.id:
            errors.append("Email already in use")

        if errors:
            for error in errors:
                flash(error, 'error')
            return redirect(url_for('admin.edit_user_form', user_id=user_id))

        user.username = username
        user.email = email
        user.is_active = is_active
        user.is_admin = is_admin

        if new_password:
            user.set_password(new_password)

        from app.models.audit_log import AuditLog
        db.session.add(AuditLog(user_id=current_user.id, table_name='users', record_id=user_id,
            change_type='UPDATE', field_name='profile', new_value=f'email={email}', ip_address=request.remote_addr))
        db.session.commit()
        flash('User updated successfully', 'success')
        return redirect(url_for('admin.edit_user_form', user_id=user_id))

    except Exception as e:
        current_app.logger.error(f"Update user error: {str(e)}\n{traceback.format_exc()}")
        flash('Error updating user', 'error')
        return redirect(url_for('admin.edit_user_form', user_id=user_id))


@admin_bp.route('/users/<int:user_id>/delete', methods=['POST'])
@require_admin
def delete_user(user_id):
    try:
        user = User.query.get_or_404(user_id)

        if not can_manage_user(user, current_user):
            return jsonify({'error': 'You cannot delete this user'}), 403

        if user.is_admin:
            from flask import current_app
            admin_count = User.query.filter_by(is_admin=True).count()

            if admin_count <= 1:
                flash('Cannot delete the last admin user', 'error')
                return redirect(url_for('admin.list_users'))

        from app.extensions import db
        from app.models.audit_log import AuditLog
        db.session.add(AuditLog(user_id=current_user.id, table_name='users', record_id=user_id,
            change_type='DELETE', field_name='username', old_value=user.username, ip_address=request.remote_addr))
        db.session.delete(user)
        db.session.commit()
        flash('User deleted successfully', 'success')

    except Exception as e:
        current_app.logger.error(f"Delete user error: {str(e)}\n{traceback.format_exc()}")
        flash('Error deleting user', 'error')

    return redirect(url_for('admin.list_users'))


@admin_bp.route('/users/<int:user_id>/reset-password', methods=['POST'])
@require_admin
def reset_password(user_id):
    try:
        success, new_password = reset_user_password(user_id)

        if not success:
            return jsonify({'error': new_password}), 404

        from app.models.audit_log import AuditLog
        db.session.add(AuditLog(user_id=current_user.id, table_name='users', record_id=user_id,
            change_type='UPDATE', field_name='password', new_value='reset', ip_address=request.remote_addr))
        db.session.commit()
        flash(f'Password reset successful. New temporary password: {new_password}', 'success')

    except Exception as e:
        current_app.logger.error(f"Reset password error: {str(e)}\n{traceback.format_exc()}")
        flash('Error resetting password', 'error')

    return redirect(url_for('admin.list_users'))


@admin_bp.route('/users/pending')
@require_admin
def pending_approvals():
    try:
        pending_users = User.query.filter_by(is_approved=False).order_by(User.created_at.desc()).all()

        return render_template('admin/pending_approvals.html', pending_users=pending_users)

    except Exception as e:
        current_app.logger.error(f"Pending approvals error: {str(e)}\n{traceback.format_exc()}")
        flash('Error loading pending approvals', 'error')
        return redirect(url_for('admin.dashboard'))


@admin_bp.route('/users/<int:user_id>/approve', methods=['POST'])
@require_admin
def approve_user(user_id):
    try:
        user = User.query.get_or_404(user_id)
        user.is_approved = True

        from app.models.audit_log import AuditLog
        db.session.add(AuditLog(user_id=current_user.id, table_name='users', record_id=user_id,
            change_type='UPDATE', field_name='is_approved', old_value='False', new_value='True', ip_address=request.remote_addr))
        db.session.commit()
        flash(f'User "{user.username}" approved', 'success')

    except Exception as e:
        current_app.logger.error(f"Approve user error: {str(e)}\n{traceback.format_exc()}")
        flash('Error approving user', 'error')

    return redirect(url_for('admin.pending_approvals'))


@admin_bp.route('/users/<int:user_id>/reject', methods=['POST'])
@require_admin
def reject_user(user_id):
    try:
        user = User.query.get_or_404(user_id)
        username = user.username

        from app.models.audit_log import AuditLog
        db.session.add(AuditLog(user_id=current_user.id, table_name='users', record_id=user_id,
            change_type='DELETE', field_name='username', old_value=username, ip_address=request.remote_addr))
        db.session.delete(user)
        db.session.commit()
        flash(f'Sign-up request for "{username}" rejected', 'success')

    except Exception as e:
        current_app.logger.error(f"Reject user error: {str(e)}\n{traceback.format_exc()}")
        flash('Error rejecting user', 'error')

    return redirect(url_for('admin.pending_approvals'))


@admin_bp.route('/users/<int:user_id>/activity')
@require_admin
def user_activity(user_id):
    try:
        user = User.query.get_or_404(user_id)

        from app.models import AuditLog

        upload_logs = AuditLog.query.filter_by(
            user_id=user_id,
            table_name='uploads'
        ).order_by(AuditLog.changed_at.desc()).all()

        return render_template('admin/user_activity.html',
                               user=user,
                               upload_logs=upload_logs)

    except Exception as e:
        current_app.logger.error(f"User activity error: {str(e)}\n{traceback.format_exc()}")
        flash('Error loading user activity', 'error')
        return redirect(url_for('admin.dashboard'))