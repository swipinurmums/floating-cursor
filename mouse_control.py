
import cv2
import mediapipe as mp
import pyautogui
import time
import math


# ----------------------------
# MOUSE CONFIGURATION
# ----------------------------

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

screen_width, screen_height = pyautogui.size()

SMOOTHING_MIN_CUTOFF = 1.0
SMOOTHING_BETA = 0.025

X_MIN = 0.25
X_MAX = 0.75

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

screen_aspect = screen_width / screen_height
x_range = X_MAX - X_MIN

y_range = (
    x_range * CAMERA_WIDTH /
    (screen_aspect * CAMERA_HEIGHT)
)

VERTICAL_OFFSET = 0.16

Y_MIN = 0.5 - y_range / 2 + VERTICAL_OFFSET
Y_MAX = 0.5 + y_range / 2 + VERTICAL_OFFSET


# ----------------------------
# PINCH AND DRAG CONFIGURATION
# ----------------------------

PINCH_START_THRESHOLD = 0.25
PINCH_RELEASE_THRESHOLD = 0.35

# How long a pinch must be held before dragging starts
DRAG_HOLD_TIME = 0.35

pinch_tracking = False
drag_active = False

pinch_start_time = None
pinch_cursor_position = None

pinch_start_filtered_x = None
pinch_start_filtered_y = None


# ----------------------------
# ONE EURO FILTER
# ----------------------------

class OneEuroFilter:

    def __init__(self, min_cutoff=1.0, beta=0.025,
                 derivative_cutoff=1.0):

        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff

        self.previous_value = None
        self.previous_derivative = 0
        self.previous_time = None

    def smoothing_factor(self, dt, cutoff):

        tau = 1.0 / (2 * math.pi * cutoff)

        return 1.0 / (1.0 + tau / dt)

    def filter(self, value, timestamp):

        if self.previous_value is None:
            self.previous_value = value
            self.previous_time = timestamp
            return value

        dt = max(timestamp - self.previous_time, 0.001)

        derivative = (
            value - self.previous_value
        ) / dt

        derivative_alpha = self.smoothing_factor(
            dt, self.derivative_cutoff
        )

        filtered_derivative = (
            derivative_alpha * derivative
            + (1 - derivative_alpha) * self.previous_derivative
        )

        cutoff = (
            self.min_cutoff
            + self.beta * abs(filtered_derivative)
        )

        alpha = self.smoothing_factor(dt, cutoff)

        filtered_value = (
            alpha * value
            + (1 - alpha) * self.previous_value
        )

        self.previous_value = filtered_value
        self.previous_derivative = filtered_derivative
        self.previous_time = timestamp

        return filtered_value

    def reset(self):
        self.previous_value = None
        self.previous_derivative = 0
        self.previous_time = None


filter_x = OneEuroFilter(
    SMOOTHING_MIN_CUTOFF, SMOOTHING_BETA
)

filter_y = OneEuroFilter(
    SMOOTHING_MIN_CUTOFF, SMOOTHING_BETA
)


# ----------------------------
# HAND TRACKING CONFIGURATION
# ----------------------------

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

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


# ----------------------------
# CAMERA
# ----------------------------

camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not camera.isOpened():
    print("Could not access webcam.")
    exit()

print("Floating Cursor active!")
print("Press Q to quit.")
print("Emergency stop: move physical mouse to top-left corner.")

try:

    with HandLandmarker.create_from_options(options) as landmarker:

        while True:

            success, frame = camera.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape

            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(time.monotonic() * 1000)

            results = landmarker.detect_for_video(
                mp_image, timestamp_ms
            )

            # Draw movement region
            cv2.rectangle(
                frame,
                (int(X_MIN * width), int(Y_MIN * height)),
                (int(X_MAX * width), int(Y_MAX * height)),
                (255, 255, 0), 2
            )

            if results.hand_landmarks:

                hand = results.hand_landmarks[0]

                points = []

                for landmark in hand:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    points.append((x, y))

                # Draw hand skeleton
                for start, end in CONNECTIONS:
                    cv2.line(
                        frame,
                        points[start],
                        points[end],
                        (0, 255, 0),
                        2
                    )

                # Draw landmark points
                for point in points:
                    cv2.circle(
                        frame, point, 5,
                        (0, 0, 255), -1
                    )

                # Highlight index fingertip
                index_tip = points[8]

                cv2.circle(
                    frame, index_tip, 10,
                    (255, 0, 0), -1
                )

                # Filter fingertip coordinates
                timestamp = time.monotonic()

                landmark = hand[8]

                filtered_x = filter_x.filter(
                    landmark.x, timestamp
                )

                filtered_y = filter_y.filter(
                    landmark.y, timestamp
                )

                # Map active camera area to screen
                target_x = (
                    (filtered_x - X_MIN) / (X_MAX - X_MIN)
                ) * (screen_width - 1)

                target_y = (
                    (filtered_y - Y_MIN) / (Y_MAX - Y_MIN)
                ) * (screen_height - 1)

                # Keep cursor inside screen
                target_x = max(
                    0, min(target_x, screen_width - 1)
                )

                target_y = max(
                    0, min(target_y, screen_height - 1)
                )

                # ----------------------------
                # PINCH DETECTION
                # ----------------------------

                thumb_tip = hand[4]
                index_finger_tip = hand[8]

                pinch_distance = math.hypot(
                    thumb_tip.x - index_finger_tip.x,
                    thumb_tip.y - index_finger_tip.y
                )

                # Normalise against palm size
                palm_size = math.hypot(
                    hand[0].x - hand[9].x,
                    hand[0].y - hand[9].y
                )

                pinch_ratio = (
                    pinch_distance / max(palm_size, 0.001)
                )

                current_time = time.monotonic()

                # ----------------------------
                # PINCH AND DRAG STATE MACHINE
                # ----------------------------

                if not pinch_tracking:

                    if pinch_ratio < PINCH_START_THRESHOLD:

                        # Start pinch and capture cursor position
                        pinch_tracking = True
                        pinch_start_time = current_time

                        pinch_cursor_position = pyautogui.position()

                        # Remember starting fingertip coordinates
                        pinch_start_filtered_x = filtered_x
                        pinch_start_filtered_y = filtered_y

                        gesture_text = "PINCHING..."
                        gesture_colour = (0, 165, 255)

                    else:

                        # Normal cursor movement
                        pyautogui.moveTo(
                            int(target_x),
                            int(target_y)
                        )

                        gesture_text = "CURSOR ACTIVE"
                        gesture_colour = (0, 255, 0)

                else:

                    # Fingers are still pinched
                    if pinch_ratio <= PINCH_RELEASE_THRESHOLD:

                        elapsed = (
                            current_time - pinch_start_time
                        )

                        if not drag_active:

                            if elapsed >= DRAG_HOLD_TIME:

                                # Begin dragging at captured position
                                pyautogui.moveTo(
                                    pinch_cursor_position.x,
                                    pinch_cursor_position.y
                                )

                                pyautogui.mouseDown()

                                drag_active = True

                        if drag_active:

                            # Calculate movement relative to
                            # where the pinch originally started
                            delta_x = (
                                (filtered_x - pinch_start_filtered_x)
                                / (X_MAX - X_MIN)
                            ) * (screen_width - 1)

                            delta_y = (
                                (filtered_y - pinch_start_filtered_y)
                                / (Y_MAX - Y_MIN)
                            ) * (screen_height - 1)

                            drag_x = (
                                pinch_cursor_position.x + delta_x
                            )

                            drag_y = (
                                pinch_cursor_position.y + delta_y
                            )

                            # Keep drag inside screen
                            drag_x = max(
                                0, min(drag_x, screen_width - 1)
                            )

                            drag_y = max(
                                0, min(drag_y, screen_height - 1)
                            )

                            pyautogui.moveTo(
                                int(drag_x),
                                int(drag_y)
                            )

                            gesture_text = "DRAGGING"
                            gesture_colour = (255, 0, 255)

                        else:

                            gesture_text = "HOLD TO DRAG"
                            gesture_colour = (0, 165, 255)

                    else:

                        # Fingers have separated
                        if drag_active:

                            # Release mouse button to drop object
                            pyautogui.mouseUp()

                            gesture_text = "DROPPED"
                            gesture_colour = (0, 255, 0)

                        else:

                            # Short pinch = normal left-click
                            pyautogui.click(
                                x=pinch_cursor_position.x,
                                y=pinch_cursor_position.y
                            )

                            gesture_text = "CLICKED"
                            gesture_colour = (0, 255, 0)

                        # Reset gesture state
                        pinch_tracking = False
                        drag_active = False

                        pinch_start_time = None
                        pinch_cursor_position = None

                        pinch_start_filtered_x = None
                        pinch_start_filtered_y = None

                        # Resume normal cursor movement
                        pyautogui.moveTo(
                            int(target_x),
                            int(target_y)
                        )

                # ----------------------------
                # VISUAL FEEDBACK
                # ----------------------------

                cv2.putText(
                    frame,
                    gesture_text,
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    gesture_colour,
                    2
                )

                cv2.line(
                    frame,
                    points[4],
                    points[8],
                    gesture_colour,
                    3
                )

                cv2.putText(
                    frame,
                    f"Index: {index_tip}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2
                )

            else:

                # If tracking is lost during a drag,
                # release the mouse button immediately
                if drag_active:
                    pyautogui.mouseUp()

                filter_x.reset()
                filter_y.reset()

                pinch_tracking = False
                drag_active = False

                pinch_start_time = None
                pinch_cursor_position = None

                pinch_start_filtered_x = None
                pinch_start_filtered_y = None

                cv2.putText(
                    frame,
                    "NO HAND DETECTED",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2
                )

            cv2.imshow("Floating Cursor", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:

    # Safety: never leave the mouse button held down
    pyautogui.mouseUp()

    camera.release()
    cv2.destroyAllWindows()
