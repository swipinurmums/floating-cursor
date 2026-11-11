
import cv2

# Connect to the laptop's default webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not access the webcam.")
    exit()

print("Webcam connected! Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read from webcam.")
        break

    # Mirror the image, like a selfie camera
    frame = cv2.flip(frame, 1)

    cv2.imshow("Floating Cursor - Camera Test", frame)

    # Press Q to close the window
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
