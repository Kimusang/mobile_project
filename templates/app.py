# $ export FLASK_APP=app.py

from flask import Flask, url_for
from flask import render_template

import os
from flask import Flask, request, jsonify, render_template
from mmdet.apis import init_detector, inference_detector

from pathlib import Path
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

# from apps.crud.models import Prediction
db = SQLAlchemy()
migrate = Migrate()

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)  # 고유 ID
    image_path = db.Column(db.String(256), nullable=True)  # 이미지 경로
    class_id = db.Column(db.Integer, nullable=True)  # 클래스 ID
    object_id = db.Column(db.Integer, nullable=True)  # 객체 ID
    x1 = db.Column(db.Float, nullable=True)  # x1 좌표
    y1 = db.Column(db.Float, nullable=True)  # y1 좌표
    x2 = db.Column(db.Float, nullable=True)  # x2 좌표
    y2 = db.Column(db.Float, nullable=True)  # y2 좌표
    score = db.Column(db.Float, nullable=True)  # 신뢰도 점수

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = Flask(__name__)

# def create_app():
    # app = Flask(__name__)

    # 앱 설정
app.config.from_mapping(
    # SECRET_KEY="2AZSMss3p5QPbcY2hBsJ",
    SQLALCHEMY_DATABASE_URI=f"sqlite:///{Path(__file__).parent / 'local.sqlite'}",
    SQLALCHEMY_TRACK_MODIFICATIONS=False
)

db.init_app(app)
migrate.init_app(app, db)

    # return app


@app.route('/showImg')
def showImg():
    return render_template('showImg.html')

@app.route('/')
def index():
    return 'http://127.0.0.1:5000/showImg', 'http://127.0.0.1:5000/predict', 'http://127.0.0.1:5000/value'


config_file = "A:\\project ai\\GRCNN-main\\GRCNN-main\\mmdetection\\configs\\ms_rcnn\\shkms_rcnn_r101_caffe_fpn_1x_coco.py"
checkpoint_file = "A:\\project ai\\GRCNN-main\\GRCNN-main\\mmdetection\\configs\\ms_rcnn\\epoch_62.pth"
model = init_detector(config_file, checkpoint_file, device="cpu")  # CPU 모드

# def register_routes(app):
#     @app.route("/showImg")
#     def showImg():
#         return render_template("showImg.html")
    
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    # 파일 저장
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    # 모델 추론
    result = inference_detector(model, file_path)

    # 결과 반환
    result_path = os.path.join(UPLOAD_FOLDER, "result_" + file.filename)
    model.show_result(file_path, result, out_file=result_path)

    # return jsonify({
    #     "message": "Prediction complete",
    #     "input_image": file_path,
    #     "result_image": result_path
    # })
    return render_template(
        "result.html",
        input_image=url_for("static", filename=f"uploads/{file.filename}"),
        result_image=url_for("static", filename=f"uploads/result_{file.filename}")
    )

@app.route("/values", methods=["GET"])
def values():
    image_path = request.args.get("image_path")
    if not image_path:
        return jsonify({"error": "No image_path provided"}), 400
# http://127.0.0.1:5000/values?image_path=static/uploads/(17).jpg


    # static 디렉토리를 기준으로 파일 시스템 경로로 변환
    full_image_path = os.path.join(app.root_path, image_path.replace("/", os.sep))

    # 파일 존재 여부 확인
    if not os.path.exists(full_image_path):
        return jsonify({"error": f"File does not exist: {full_image_path}"}), 404

    # 모델 예측 수행
    result = inference_detector(model, full_image_path)

    # 결과 구조 확인
    if isinstance(result, tuple):
        bbox_result, _ = result
    else:
        bbox_result = result

    output = []
    for i, bboxes in enumerate(bbox_result):
        class_results = {"class": i, "objects": []}
        for j, bbox in enumerate(bboxes):
            if len(bbox) == 5:
                x1, y1, x2, y2, score = bbox
                if score >= 0.3:  # 임계값 0.3 이상인 경우만 포함
                    class_results["objects"].append({
                        "object_id": j + 1,
                        "coordinates": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                        "score": float(score)
                    })
            else:
                class_results["objects"].append({"error": "Unexpected bbox format"})
        output.append(class_results)

    return jsonify(output)

@app.route("/value", methods=["GET"])
def value():
    image_path = request.args.get("image_path")
    if not image_path:
        return jsonify({"error": "No image_path provided"}), 400
# # http://127.0.0.1:5000/value?image_path=static/uploads/(17).jpg
#     # static 디렉토리를 기준으로 파일 시스템 경로로 변환
#     full_image_path = os.path.join(app.root_path, image_path.replace("/", os.sep))
    file_name = image_path.replace("/static/uploads/", "")
    
    # 실제 파일 경로로 변환
    full_image_path = os.path.join(app.root_path, 'static', 'uploads', file_name)
    # 파일 존재 여부 확인
    if not os.path.exists(full_image_path):
        return jsonify({"error": f"File does not exist: {full_image_path}"}), 404

    # 모델 예측 수행
    result = inference_detector(model, full_image_path)

    # 결과 구조 확인
    if isinstance(result, tuple):
        bbox_result, _ = result
    else:
        bbox_result = result

    # 데이터 저장
    for class_id, bboxes in enumerate(bbox_result):
        for object_id, bbox in enumerate(bboxes):
            if len(bbox) == 5:
                x1, y1, x2, y2, score = bbox
                if score >= 0.3:  # 임계값 0.3 이상만 저장
                    prediction = Prediction(
                        image_path=image_path,
                        class_id=class_id,
                        object_id=object_id + 1,
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        score=float(score)
                    )
                    db.session.add(prediction)
    db.session.commit()

    return jsonify({"message": "Results saved to database."})

if __name__ == '__main__' :
    # app.run(host = '0.0.0.0', port = 80, debug = True)
    app.run(debug = True)
