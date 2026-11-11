
import cv2
import mediapipe as mp
import time

# MediaPipe hand-tracking setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

# Connections between the 21 hand landmarks
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not access webcam.")
    exit()

print("Hand tracking started! Press Q to quit.")

try:
    with HandLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = camera.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)

            # Convert OpenCV image to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Process the video frame
            timestamp_ms = int(time.monotonic() * 1000)
            results = landmarker.detect_for_video(
                mp_image, timestamp_ms
            )

            height, width, _ = frame.shape

            if results.hand_landmarks:
                for hand in results.hand_landmarks:

                    points = []

                    for landmark in hand:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        points.append((x, y))

                    # Draw connections between landmarks
                    for start, end in connections:
                        cv2.line(
                            frame,
                            points[start],
                            points[end],
                            (0, 255, 0),
                            2
                        )

                    # Draw all 21 landmarks
                    for point in points:
                        cv2.circle(
                            frame, point, 5,
                            (0, 0, 255), -1
                        )

                    # Highlight index fingertip (landmark 8)
                    index_tip = points[8]

                    cv2.circle(
                        frame, index_tip, 10,
                        (255, 0, 0), -1
                    )

                    cv2.putText(
                        frame,
                        f"Index: {index_tip}",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                cv2.putText(
                    frame, "HAND DETECTED",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )

            else:
                cv2.putText(
                    frame, "No hand detected",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2
                )

            cv2.imshow("Floating Cursor - Hand Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
