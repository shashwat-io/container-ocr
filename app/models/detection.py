from datetime import datetime
from sqlalchemy import Column, Integer, Float, DateTime, Text, ForeignKey, String, Boolean
from sqlalchemy.orm import relationship
from app.extensions import db

class Detection(db.Model):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True)
    upload_id = Column(Integer, ForeignKey("uploads.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    crop_path = Column(String(255), nullable=True)
    container_code = Column(String(20), nullable=True)
    is_valid = Column(Boolean, nullable=True)
    confidence_score = Column(Float, nullable=False)
    is_flagged = Column(Boolean, default=False, nullable=False)
    is_edited = Column(Boolean, default=False, nullable=False)
    ocr_raw_output = Column(Text, nullable=True)
    ocr_corrected = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    notes = Column(Text, nullable=True)
    upload = relationship("Upload", back_populates="detections")
    user = relationship("User", foreign_keys=[user_id])

    def __repr__(self):
        return f"<Detection {self.id}>"
