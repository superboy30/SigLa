from start import *


model = Sequential()

model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

# 1. New detection variables

model.load_weights('action.h5') 

# 1. New detection variables
sequence = []
sentence = ''
predictions = []
threshold = 0.5

cap = cv2.VideoCapture(1)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read() 

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            # print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            
        #3. Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    # If you want a subtitle type of text 
                    sentence = actions[np.argmax(res)];
                    
                    # if len(sentence) > 0: 
                    #     if actions[np.argmax(res)] != sentence[-1]:
                    #         sentence.append(actions[np.argmax(res)])
                    # else:
                    #     sentence.append(actions[np.argmax(res)])

            # if len(sentence) > 5: 
            #     sentence = sentence[-5:]

            # Viz probabilities
            # image = prob_viz(res, actions, image, colors)
            
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        textsize = cv2.getTextSize(sentence, font, 1, 2)[0]
        
        textX = (image.shape[1] - textsize[0]) * 0.5
        textY = (image.shape[0] + textsize[1])
        
        if(np.all(keypoints==0) == False):
            cv2.rectangle(image,  (0, int(textY)), (640, int(textY) - 80), (0, 0, 0), -1)
            cv2.putText(image, sentence, (int(textX), int(textY) - 40 ), font, 1, (0, 0, 255), 2)
        
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()