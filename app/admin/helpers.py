from werkzeug.security import generate_password_hash
from app.models import User


def validate_user_input(username, email, password=None):
    errors = []

    if not username or len(username.strip()) < 3:
        errors.append("Username must be at least 3 characters long")

    if not email or "@" not in email:
        errors.append("Valid email is required")

    if password:
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")

    return errors


def reset_user_password(user_id, new_password=None):
    user = User.query.get(user_id)
    if not user:
        return False, "User not found"

    if new_password:
        user.set_password(new_password)
    else:
        import random
        import string

        new_password = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
        user.set_password(new_password)

    return True, new_password


def generate_temporary_password(length=12):
    import random
    import string

    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def can_manage_user(target_user, current_user):
    if current_user.id == target_user.id:
        return False

    if not current_user.is_admin:
        return False

    return True