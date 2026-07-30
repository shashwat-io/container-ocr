import click
from flask.cli import with_appcontext
from sqlalchemy.exc import OperationalError, IntegrityError

from app.extensions import db
from app.models.user import User


@click.command("create-user")
@click.option("--username", prompt=True, help="Unique username for the new user.")
@click.option("--email", prompt=True, help="Unique email address for the new user.")
@click.option(
    "--password",
    prompt=True,
    hide_input=True,
    confirmation_prompt=True,
    help="Password for the new user (prompted securely if omitted).",
)
@click.option("--admin", is_flag=True, default=False, help="Grant admin privileges to the new user.")
@with_appcontext
def create_user_command(username, email, password, admin):
    """Create a new user account (bootstrap / admin-only account creation)."""
    username = username.strip()
    email = email.strip().lower()

    if not username:
        raise click.ClickException("Username cannot be empty.")
    if not email or "@" not in email:
        raise click.ClickException("A valid email address is required.")
    if not password:
        raise click.ClickException("Password cannot be empty.")

    try:
        existing = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()
    except OperationalError as exc:
        raise click.ClickException(
            "Could not query the database (is the schema migrated? run "
            "'flask db upgrade' first). Make sure you're running this command "
            "from the project root so the Flask app and its config load "
            f"correctly.\nUnderlying error: {exc}"
        ) from exc

    if existing:
        if existing.username == username:
            raise click.ClickException(f'Username "{username}" is already taken.')
        raise click.ClickException(f'Email "{email}" is already in use.')

    user = User(username=username, email=email, is_active=True, is_admin=admin)
    user.set_password(password)

    try:
        db.session.add(user)
        db.session.commit()
    except (OperationalError, IntegrityError) as exc:
        db.session.rollback()
        raise click.ClickException(f"Failed to create user in the database: {exc}") from exc

    role = " (admin)" if admin else ""
    click.echo(f'Created user successfully: username="{username}", email="{email}"{role}')


def register_cli(app):
    app.cli.add_command(create_user_command)
