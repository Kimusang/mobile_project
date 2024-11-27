# $ export FLASK_APP=app.py
# flask db migrate -m "Describe the changes made"
# flask db upgrade
from flask import Flask, url_for
from flask import render_template

import os
from flask import Flask, request, jsonify, render_template
from mmdet.apis import init_detector, inference_detector

from pathlib import Path
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

import matplotlib
matplotlib.use('Agg')

# from apps.crud.models import Prediction
db = SQLAlchemy()
migrate = Migrate()

# class Prediction(db.Model):
#     id = db.Column(db.Integer, primary_key=True)  # 고유 ID
#     image_path = db.Column(db.String(256), nullable=True)  # 이미지 경로
#     class_id = db.Column(db.Integer, nullable=True)  # 클래스 ID
#     object_id = db.Column(db.Integer, nullable=True)  # 객체 ID
#     x1 = db.Column(db.Float, nullable=True)  # x1 좌표
#     y1 = db.Column(db.Float, nullable=True)  # y1 좌표
#     x2 = db.Column(db.Float, nullable=True)  # x2 좌표
#     y2 = db.Column(db.Float, nullable=True)  # y2 좌표
#     score = db.Column(db.Float, nullable=True)  # 신뢰도 점수

class Prediction1(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(256), nullable=False)
    class_id = db.Column(db.Integer, nullable=True)
    object_id = db.Column(db.Integer, nullable=True)
    x1 = db.Column(db.Float, nullable=True)
    y1 = db.Column(db.Float, nullable=True)
    x2 = db.Column(db.Float, nullable=True)
    y2 = db.Column(db.Float, nullable=True)
    score = db.Column(db.Float, nullable=True)

class Prediction2(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(256), nullable=False)
    class_id = db.Column(db.Integer, nullable=True)
    object_id = db.Column(db.Integer, nullable=True)
    x1 = db.Column(db.Float, nullable=True)
    y1 = db.Column(db.Float, nullable=True)
    x2 = db.Column(db.Float, nullable=True)
    y2 = db.Column(db.Float, nullable=True)
    score = db.Column(db.Float, nullable=True)

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
    return jsonify({
        "routes": {
            "/showImg": "Render uploaded images",
            "/predict": "POST an image for prediction",
            "/values": "GET bounding box details",
            "/value": "Save bounding box results to DB",
            "/model1_db": "Search result values ​​by image name from DB"
        }
    })

# config_file = "A:\\project ai\\GRCNN-main\\GRCNN-main\\mmdetection\\configs\\ms_rcnn\\shkms_rcnn_r101_caffe_fpn_1x_coco.py"
# checkpoint_file = "A:\\project ai\\GRCNN-main\\GRCNN-main\\mmdetection\\configs\\ms_rcnn\\epoch_62.pth"
# model = init_detector(config_file, checkpoint_file, device="cpu")  # CPU 모드
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL1_CONFIG = os.path.join(BASE_DIR, "configs", "ms_rcnn", "shkms_rcnn_r101_caffe_fpn_1x_coco.py")
MODEL1_CHECKPOINT = os.path.join(BASE_DIR, "configs", "ms_rcnn", "epoch_62.pth")
MODEL2_CONFIG = os.path.join(BASE_DIR, "configs", "ms_rcnn", "ms_rcnn_r50_caffe_fpn_1x_coco_caries.py")
MODEL2_CHECKPOINT = os.path.join(BASE_DIR, "configs", "ms_rcnn", "latest.pth")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 모델 초기화
model = init_detector(MODEL1_CONFIG, MODEL1_CHECKPOINT, device="cpu")
model_ca = init_detector(MODEL2_CONFIG, MODEL2_CHECKPOINT, device="cpu")
# def register_routes(app):
#     @app.route("/showImg")
#     def showImg():
#         return render_template("showImg.html")
    
@app.route("/predict", methods=["POST"])   # 모델 실행 결과 이미지
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # 파일 저장
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # 모델 1 추론
    result1 = inference_detector(model, file_path)
    result1_path = os.path.join(UPLOAD_FOLDER, f"result1_{file.filename}")
    model.show_result(file_path, result1, out_file=result1_path)

    # 모델 2 추론
    result2 = inference_detector(model_ca, file_path)
    result2_path = os.path.join(UPLOAD_FOLDER, f"result2_{file.filename}")
    model_ca.show_result(file_path, result2, out_file=result2_path)

    # 결과 페이지 렌더링
    model1_results = process_results(result1)  # 결과 처리 함수
    model2_results = process_results(result2)  # 결과 처리 함수

    return render_template(
        "result.html",
        input_image=url_for("static", filename=f"uploads/{file.filename}"),
        result1_image=url_for("static", filename=f"uploads/result1_{file.filename}"),
        result2_image=url_for("static", filename=f"uploads/result2_{file.filename}"),
        model1_results=model1_results,
        model2_results=model2_results
    )

# @app.route("/values_Ca", methods=["GET"])
# def values_Ca():
#     image_path = request.args.get("image_path")
#     if not image_path:
#         return jsonify({"error": "No image_path provided"}), 400
# # http://127.0.0.1:5000/values_Ca?image_path=static/uploads/(17).jpg


#     # static 디렉토리를 기준으로 파일 시스템 경로로 변환
#     full_image_path = os.path.join(app.root_path, image_path.replace("/", os.sep))

#     # 파일 존재 여부 확인
#     if not os.path.exists(full_image_path):
#         return jsonify({"error": f"File does not exist: {full_image_path}"}), 404

#     # 모델 예측 수행
#     result = inference_detector(model_ca, full_image_path)

#     # 결과 구조 확인
#     if isinstance(result, tuple):
#         bbox_result, _ = result
#     else:
#         bbox_result = result

#     output = []
#     for i, bboxes in enumerate(bbox_result):
#         class_results = {"class": i, "objects": []}
#         for j, bbox in enumerate(bboxes):
#             if len(bbox) == 5:
#                 x1, y1, x2, y2, score = bbox
#                 if score >= 0.3:  # 임계값 0.3 이상인 경우만 포함
#                     class_results["objects"].append({
#                         "object_id": j + 1,
#                         "coordinates": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
#                         "score": float(score)
#                     })
#             else:
#                 class_results["objects"].append({"error": "Unexpected bbox format"})
#         output.append(class_results)

#     return jsonify(output)


@app.route("/values", methods=["GET"])   # 모델의 값들을 보기
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


@app.route("/value", methods=["POST"])   # 모델들의 값을 저장
def value():
    data = request.get_json()
    image_path = data.get("image_path")
    model_results = data.get("model_results")

    if not image_path or not model_results:
        return jsonify({"error": "Invalid data provided"}), 400

    # Save results for Model 1
    if "model1" in model_results:
        for bbox in model_results["model1"]:
            prediction = Prediction1(
                image_path=image_path,
                class_id=bbox["class_id"],
                object_id=bbox["object_id"],
                x1=bbox["x1"],
                y1=bbox["y1"],
                x2=bbox["x2"],
                y2=bbox["y2"],
                score=bbox["score"]
            )
            db.session.add(prediction)

    # Save results for Model 2
    if "model2" in model_results:
        for bbox in model_results["model2"]:
            prediction = Prediction2(
                image_path=image_path,
                class_id=bbox["class_id"],
                object_id=bbox["object_id"],
                x1=bbox["x1"],
                y1=bbox["y1"],
                x2=bbox["x2"],
                y2=bbox["y2"],
                score=bbox["score"]
            )
            db.session.add(prediction)

    # Commit the session to save to the database
    db.session.commit()

    return jsonify({"message": "Results saved to database."})

@app.route('/model1_db', methods=['GET', 'POST'])
def model1_db():
    if request.method == 'POST':
        img_name = request.form['img_name']
        # Searching for entries where image_path contains the img_name
        results = Prediction1.query.filter(Prediction1.image_path.like(f'%{img_name}%')).all()
        return render_template('model_db.html', results=results, img_name=img_name)
    return render_template('model_db.html', results=None)

@app.route('/model2_db', methods=['GET', 'POST'])
def model2_db():
    if request.method == 'POST':
        img_name = request.form['img_name']
        # Searching for entries where image_path contains the img_name
        results = Prediction2.query.filter(Prediction2.image_path.like(f'%{img_name}%')).all()
        return render_template('model_db.html', results=results, img_name=img_name)
    return render_template('model_db.html', results=None)


def process_results(result):     # 결과 처리 함수
    output = []
    bbox_result = result[0] if isinstance(result, tuple) else result

    for class_id, bboxes in enumerate(bbox_result):
        for object_id, bbox in enumerate(bboxes):
            if len(bbox) == 5:
                x1, y1, x2, y2, score = bbox
                if score >= 0.3:
                    output.append({
                        "class_id": class_id,
                        "object_id": object_id + 1,
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "score": float(score)
                    })
    return output




if __name__ == '__main__' :
    # app.run(host = '0.0.0.0', port = 80, debug = True)
    app.run(debug = True)
