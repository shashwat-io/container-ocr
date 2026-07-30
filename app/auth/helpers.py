from werkzeug.security import check_password_hash
from app.models.user import User


def verify_user_credentials(username, password):
    user = User.query.filter_by(username=username).first()

    if user and check_password_hash(user.password_hash, password):
        return user

    return None