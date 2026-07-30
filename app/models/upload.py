from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from app.extensions import db

class Upload(db.Model):
    __tablename__ = "uploads"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    filename = Column(String(255), nullable=False)
    image_size = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    is_edited = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)
    detections = relationship("Detection", back_populates="upload", cascade="all, delete-orphan")
    user = relationship("User", back_populates="uploads")

    def __repr__(self):
        return f"<Upload {self.filename}>"
