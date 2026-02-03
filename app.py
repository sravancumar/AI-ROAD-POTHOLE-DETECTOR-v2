from flask import Flask, request, render_template
from ultralytics import YOLO
from geopy.geocoders import Nominatim
import os, uuid, cv2

app = Flask(__name__)
model = YOLO("pothole_guard.onnx")

# Use absolute paths to ensure it works on Windows and Vercel
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

geolocator = Nominatim(user_agent="air_pothole_v8_final")

@app.route("/")
def home(): return render_template("index.html")

@app.route("/history")
def history(): return render_template("history.html")

@app.route("/complaint")
def complaint(): return render_template("complaint.html")

@app.route("/detect_multiple", methods=["POST"])
def detect_multiple():
    files = request.files.getlist("images")
    lat, lon = request.form.get("lat"), request.form.get("lon")
    all_results, total_potholes = [], 0

    for file in files:
        if not file: continue
        unique_id = uuid.uuid4().hex
        ext = file.filename.split('.')[-1].lower()
        path = os.path.normpath(os.path.join(UPLOAD_FOLDER, f"{unique_id}.{ext}"))
        file.save(path)

        if ext in ['mp4', 'mov', 'avi']:
            cap = cv2.VideoCapture(path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 30 == 0:
                    tmp_p = os.path.join(UPLOAD_FOLDER, "frame.jpg")
                    cv2.imwrite(tmp_p, frame)
                    res = model(tmp_p, conf=0.25)
                    total_potholes += len(res[0].boxes)
                    out_name = f"v_{uuid.uuid4().hex}.jpg"
                    cv2.imwrite(os.path.join(RESULT_FOLDER, out_name), res[0].plot())
                    all_results.append(out_name)
            cap.release()
        else:
            res = model(path, conf=0.25)
            total_potholes += len(res[0].boxes)
            out_name = f"r_{unique_id}.jpg"
            cv2.imwrite(os.path.join(RESULT_FOLDER, out_name), res[0].plot())
            all_results.append(out_name)

    address = "Location Found"
    try:
        if lat and lon:
            loc = geolocator.reverse(f"{lat},{lon}", timeout=5)
            address = loc.address if loc else f"{lat}, {lon}"
    except: address = f"{lat}, {lon}"

    return render_template("result.html", potholes=total_potholes, address=address, lat=lat, lon=lon, images=all_results)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)