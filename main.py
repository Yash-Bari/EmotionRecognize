import cv2
import numpy as np
import tensorflow
from keras.models import load_model
model = load_model("emo_2.0.h5")
emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}
def detect_emotions():
    # Open the webcam feed
    cap = cv2.VideoCapture(0)

    # Define the face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Loop through the frames of the webcam feed
    while True:
        # Read a frame from the webcam feed
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Loop through the detected faces
        for (x, y, w, h) in faces:
            # Extract the face ROI
            roi = gray[y:y+h, x:x+w]

            # Resize the ROI to match the input size of the model
            roi = cv2.resize(roi, (48, 48))

            # Normalize the ROI
            roi = roi / 255.0

            # Reshape the ROI to match the input shape of the model
            roi = np.reshape(roi, (1, 48, 48, 1))

            # Predict the emotion from the ROI using the pre-trained model
            predictions = model.predict(roi)

            # Get the index of the predicted emotion
            predicted_emotion = np.argmax(predictions)

            # Get the label of the predicted emotion
            predicted_label = emotion_labels[predicted_emotion]

            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Display the predicted emotion label on the output window
            cv2.putText(frame, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame on the output window
        cv2.imshow('Emotion Detection', frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the resources
    cap.release()
    cv2.destroyAllWindows()

detect_emotions()
