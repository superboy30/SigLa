import base64
import io
from PIL import Image
# from pyimagesearch.motion_detection.singlemotiondetector import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask, jsonify
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
from ai import GestureDetector
from flask_socketio import SocketIO, emit
from io import StringIO
import numpy as np

import os

outputFrame = None
lock = threading.Lock()

app = Flask(name)
socketio = SocketIO(app)


md = GestureDetector()

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on('image')
def image(data_image):
    sbuf = StringIO()
    sbuf.write(data_image)

    # decode and convert into image
    b = io.BytesIO(base64.b64decode(data_image))
    pimg = Image.open(b)

    ## converting RGB to BGR, as opencv standards
    frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
    # Process the image frame
    frame = imutils.resize(frame, width=700)
    frame = cv2.flip(frame, 1)

    with md.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
         # Make detections
        image, results = md.mediapipe_detection(frame, holistic)
        
        # Draw landmarks
        md.draw_styled_landmarks(image, results)


        # base64 encode
        imgencode = cv2.imencode('.jpg', image)[1]

        stringData = base64.b64encode(imgencode).decode('utf-8')
        b64_src = 'data:image/jpg;base64,'
        stringData = b64_src + stringData

        # emit the frame back
        emit('response_back', stringData)


def detect_motion():

    
    global vs, outputFrame, lock

    md = GestureDetector()
    total = 0

    with md.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:

            frame = vs.read()

            # Make detections
            image, results = md.mediapipe_detection(frame, holistic)
            
            # Draw landmarks
            md.draw_styled_landmarks(image, results)

            with lock:
                outputFrame = image

def generate():

    global outputFrame, lock
    while True:
    
        with lock:
            
            if outputFrame is None:
                continue
            
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            if not flag:
                continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    return Response(
        generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame",
    )
    

if name == 'main':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
        help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
        help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
        help="# of frames used to construct the background model")
    args = vars(ap.parse_args())
    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_motion, args=(
        args["frame_count"],))
    t.daemon = True
    t.start()

    app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False, ssl_context='adhoc')
