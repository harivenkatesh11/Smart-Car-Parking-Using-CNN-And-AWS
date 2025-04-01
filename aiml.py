from flask import Flask, render_template, jsonify
import cv2
import pickle
import cvzone
import numpy as np
import threading

app = Flask(__name__)

# Load video
cap = cv2.VideoCapture(r"C:\Users\CH.HARI VENKATESH\Downloads\carPark.mp4")

# Load saved parking slot positions
with open(r"CarParkPos", 'rb') as f:
    posList = pickle.load(f)

width, height = 107, 48

# Global parking status
parking_status = {"total": len(posList), "empty": 0, "occupied": len(posList)}
free_slots = []  # Store free slot indices


def process_video():
    global parking_status, free_slots

    while True:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        success, img = cap.read()
        if not success:
            continue

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
        imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 25, 16)
        imgMedian = cv2.medianBlur(imgThreshold, 5)
        kernel = np.ones((3, 3), np.uint8)
        imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

        empty_count = 0
        free_slots.clear()

        for idx, pos in enumerate(posList):
            x, y = pos
            imgCrop = imgDilate[y:y + height, x:x + width]
            count = cv2.countNonZero(imgCrop)

            if count < 900:
                empty_count += 1
                free_slots.append(idx)
                color = (0, 255, 0)
                thickness = 5
            else:
                color = (0, 0, 255)
                thickness = 2

            cv2.rectangle(img, (x, y), (x + width, y + height), color, thickness)
            cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1,
                               thickness=2, offset=0, colorR=color)

        occupied_count = len(posList) - empty_count
        parking_status = {"total": len(posList), "empty": empty_count, "occupied": occupied_count}

        cvzone.putTextRect(img, f'Free: {empty_count}/{len(posList)}', (100, 50), scale=3,
                           thickness=5, offset=20, colorR=(0, 200, 0))

        cv2.imshow("Parking Lot", img)
        cv2.waitKey(10)


# Start video processing in a background thread
thread = threading.Thread(target=process_video, daemon=True)
thread.start()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/parking_status')
def parking_status_api():
    return jsonify(parking_status)


@app.route('/view_slots')
def view_slots():
    slot_status = ["occupied"] * len(posList)
    for idx in free_slots:
        slot_status[idx] = "empty"

    positions = [(x, y) for x, y in posList]
    return render_template('slots.html', slot_status=slot_status, positions=positions)


if __name__ == "__main__":
    app.run(debug=True)