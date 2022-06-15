

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
