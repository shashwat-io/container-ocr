from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship

from app.extensions import db


class AuditLog(db.Model):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    table_name = Column(String(50), nullable=False)
    record_id = Column(Integer, nullable=False)
    change_type = Column(String(10), nullable=False)
    field_name = Column(String(100), nullable=False)
    old_value = Column(Text)
    new_value = Column(Text)
    changed_at = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String(50))

    user = relationship("User", back_populates="audit_logs")

    def __repr__(self):
        return f"<AuditLog {self.id}>"
