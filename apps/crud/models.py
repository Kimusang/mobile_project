from datetime import datetime
# from app import db
from werkzeug.security import generate_password_hash

# class Prediction(db.Model):
#     id = db.Column(db.Integer, primary_key=True)  # 고유 ID
#     image_path = db.Column(db.String(256), nullable=False)  # 이미지 경로
#     class_id = db.Column(db.Integer, nullable=False)  # 클래스 ID
#     object_id = db.Column(db.Integer, nullable=False)  # 객체 ID
#     x1 = db.Column(db.Float, nullable=False)  # x1 좌표
#     y1 = db.Column(db.Float, nullable=False)  # y1 좌표
#     x2 = db.Column(db.Float, nullable=False)  # x2 좌표
#     y2 = db.Column(db.Float, nullable=False)  # y2 좌표
#     score = db.Column(db.Float, nullable=False)  # 신뢰도 점수
