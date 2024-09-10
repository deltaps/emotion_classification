import cv2
import os

emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

dataset_dir = 'dataset_custom'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Create two folders for training and validation
train_dir = os.path.join(dataset_dir, 'train')
validation_dir = os.path.join(dataset_dir, 'validation')

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(validation_dir):
    os.makedirs(validation_dir)

for emotion in emotions:
    emotion_train_dir = os.path.join(train_dir, emotion)
    emotion_validation_dir = os.path.join(validation_dir, emotion)

    if not os.path.exists(emotion_train_dir):
        os.makedirs(emotion_train_dir)
    
    if not os.path.exists(emotion_validation_dir):
        os.makedirs(emotion_validation_dir)

def process_image(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    # Process the first detected face
    for (x, y, w, h) in faces:
        # Crop the frame to the detected face
        face = image[y:y+h, x:x+w]

        # Replace the original image with the cropped face, and save it
        return face

def capture_photos(emotion, num_photos_train=100, num_photos_validation=20):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la webcam.")
        return

    print(f"Veuillez effectuer l'émotion '{emotion}'. Appuyez sur 's' pour commencer la capture des images.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur: Impossible de lire l'image depuis la webcam.")
            break

        cv2.imshow(f"Capture pour '{emotion}'", frame)

        # Appuyez sur 's' pour commencer la capture des photos
        if cv2.waitKey(1) & 0xFF == ord('s'):
            for i in range(num_photos_train + num_photos_validation):
                ret, frame = cap.read()
                if not ret:
                    print("Erreur: Impossible de lire l'image depuis la webcam.")
                    break
                
                frame = process_image(frame)

                if i < num_photos_train:
                    save_dir = os.path.join(train_dir, emotion)
                else:
                    save_dir = os.path.join(validation_dir, emotion)
                
                cv2.imwrite(os.path.join(save_dir, f"{emotion}_{i}.jpg"), frame)
                print(f"Image {i+1} enregistrée pour l'émotion '{emotion}'.")
            break

    cap.release()
    cv2.destroyAllWindows()

for emotion in emotions:
    capture_photos(emotion)
    print(f"Capture terminée pour l'émotion '{emotion}'. Passez à l'émotion suivante.")

print("Toutes les captures sont terminées.")