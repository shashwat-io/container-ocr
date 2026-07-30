from typing import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = 'XXXXXXXXXXXX'
down_revision = None
branch_labels = None
delete_orphan = None


def upgrade() -> None:
    op.create_table('users',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('username', sa.String(length=80), nullable=False),
    sa.Column('email', sa.String(length=120), nullable=False),
    sa.Column('password_hash', sa.String(length=256), nullable=False),
    sa.Column('is_active', sa.Boolean(), server_default='true'),
    # Was server_default='true' -- any row inserted with is_admin omitted
    # entirely (a raw INSERT bypassing the ORM, since every User() call site
    # in the app explicitly sets is_admin one way or the other -- confirmed
    # 29-07-2026 by checking cli.py, auth/routes.py signup, and
    # admin/routes.py create_user, plus a repo-wide search for any raw SQL
    # user insert) would silently become an admin. No live row is actually
    # attributable to this -- the 3 existing admins were each created via an
    # explicit is_admin=True at the ORM call site -- so this is a closed
    # footgun for future/external inserts, not a fix for existing data.
    #
    # NOTE: editing this file does NOT retroactively change the live
    # database. This migration already ran; SQLite baked server_default
    # into the users table at CREATE TABLE time and has no ALTER COLUMN
    # DEFAULT support, so the on-disk default stays 'true' until the table
    # is rebuilt. This edit only takes effect for a database built fresh
    # from `flask db upgrade` (e.g. a new deployment or test environment).
    sa.Column('is_admin', sa.Boolean(), server_default='false'),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_users_email', 'users', ['email'], unique=True)
    op.create_index('ix_users_username', 'users', ['username'], unique=True)


def downgrade() -> None:
    op.drop_index('ix_users_email', table_name='users')
    op.drop_index('ix_users_username', table_name='users')
    op.drop_table('users')
