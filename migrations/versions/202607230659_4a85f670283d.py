"""add is_approved to users

Revision ID: 4a85f670283d
Revises: f33e1542bc0a
Create Date: 2026-07-23 06:59:36.696991+00:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '4a85f670283d'
down_revision = 'f33e1542bc0a'
branch_labels = None
depends_on = None


def upgrade():
    # NOTE: autogenerate also proposed unrelated changes to detections.id,
    # detections.created_at, and uploads.user_id nullability -- those are
    # pre-existing model/live-DB drift documented in f33e1542bc0a and are
    # deliberately left untouched here; this migration only adds is_approved.
    #
    # server_default='1' backfills every pre-existing row as approved
    # (grandfathered) since they predate the approval feature; the app-level
    # default for *new* rows is controlled by the ORM (User.is_approved
    # default=True for admin/CLI-created accounts, explicitly overridden to
    # False by the public sign-up route).
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.add_column(sa.Column('is_approved', sa.Boolean(), nullable=False, server_default=sa.text('1')))


def downgrade():
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_column('is_approved')
