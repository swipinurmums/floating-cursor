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
SMOOTHING_BETA = 0.06

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

DRAG_HOLD_TIME = 0.35

pinch_tracking = False
drag_active = False

pinch_start_time = None
pinch_cursor_position = None

pinch_start_filtered_x = None
pinch_start_filtered_y = None


# ----------------------------
# RIGHT-CLICK CONFIGURATION
# ----------------------------

# Middle finger raised, with the other three fingers folded and thumb extended.
# Hold the pose briefly to avoid accidental right-clicks.
RIGHT_CLICK_HOLD_TIME = 0.4
RIGHT_CLICK_EXIT_GRACE_FRAMES = 4

right_click_start_time = None
right_click_latched = False
right_click_exit_counter = 0


# ----------------------------
# PAUSE / RESUME CONFIGURATION
# ----------------------------

# Hold an open palm for this long to toggle pause.
OPEN_PALM_HOLD_TIME = 0.8

# Require the open-palm gesture to disappear for several frames
# before allowing another pause/resume toggle.
PAUSE_EXIT_GRACE_FRAMES = 5

paused = False
pause_gesture_start = None
pause_gesture_latched = False
pause_exit_counter = 0


# ----------------------------
# SCROLL CONFIGURATION
# ----------------------------

# Large, centred scroll area
SCROLL_X_MIN = 0.05
SCROLL_X_MAX = 0.95
SCROLL_Y_MIN = 0.08
SCROLL_Y_MAX = 0.92

# Centre of the scroll zone
SCROLL_CENTRE = 0.50

# Distance from centre before scrolling starts
SCROLL_DEAD_ZONE = 0.04

# Maximum scrolling speed in wheel units per second
SCROLL_MAX_SPEED = 3500.0

# No momentum/inertia.
SCROLL_FRICTION = 0

# Allow brief tracking mistakes without exiting scroll mode
SCROLL_EXIT_GRACE_FRAMES = 3

scroll_mode = False
scroll_exit_counter = 0

scroll_velocity = 0.0
scroll_accumulator = 0.0


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
# FINGER EXTENSION DETECTION
# ----------------------------

def finger_extended(hand, tip_index, pip_index):

    wrist = hand[0]

    tip_distance = math.hypot(
        hand[tip_index].x - wrist.x,
        hand[tip_index].y - wrist.y
    )

    pip_distance = math.hypot(
        hand[pip_index].x - wrist.x,
        hand[pip_index].y - wrist.y
    )

    return tip_distance > pip_distance * 1.12


def thumb_extended(hand, palm_size):

    # Thumb tip (4) should be clearly separated from index MCP (5).
    thumb_spread = math.hypot(
        hand[4].x - hand[5].x,
        hand[4].y - hand[5].y
    )

    return thumb_spread > max(palm_size * 0.45, 0.025)


def open_palm_detected(hand, palm_size):

    return (
        thumb_extended(hand, palm_size)
        and finger_extended(hand, 8, 6)
        and finger_extended(hand, 12, 10)
        and finger_extended(hand, 16, 14)
        and finger_extended(hand, 20, 18)
    )


# ----------------------------
# SCROLL SPEED CALCULATION
# ----------------------------

def calculate_scroll_speed(hand_y):

    # Clamp position to the scroll zone
    hand_y = max(
        SCROLL_Y_MIN,
        min(hand_y, SCROLL_Y_MAX)
    )

    # Positive velocity means scroll upwards.
    displacement = SCROLL_CENTRE - hand_y

    # Ignore small movements around the centre.
    if abs(displacement) <= SCROLL_DEAD_ZONE:
        return 0.0

    # Remove the dead-zone distance.
    if displacement > 0:
        effective_distance = displacement - SCROLL_DEAD_ZONE
    else:
        effective_distance = displacement + SCROLL_DEAD_ZONE

    available_range = (
        (SCROLL_Y_MAX - SCROLL_Y_MIN) / 2
        - SCROLL_DEAD_ZONE
    )

    intensity = min(
        abs(effective_distance) / available_range,
        1.0
    )

    # Non-linear acceleration gives finer control near the centre.
    intensity = intensity ** 1.5

    speed = intensity * SCROLL_MAX_SPEED

    return math.copysign(speed, displacement)


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

display_fps = 0.0
previous_frame_time = time.monotonic()


# ----------------------------
# MAIN PROGRAM
# ----------------------------

try:

    with HandLandmarker.create_from_options(options) as landmarker:

        while True:

            frame_start_time = time.monotonic()

            dt = min(
                frame_start_time - previous_frame_time,
                0.1
            )

            previous_frame_time = frame_start_time

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

                # Draw landmarks
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

                filtered_x = filter_x.filter(
                    hand[8].x, timestamp
                )

                filtered_y = filter_y.filter(
                    hand[8].y, timestamp
                )

                # Map active camera area to screen
                target_x = (
                    (filtered_x - X_MIN) / (X_MAX - X_MIN)
                ) * (screen_width - 1)

                target_y = (
                    (filtered_y - Y_MIN) / (Y_MAX - Y_MIN)
                ) * (screen_height - 1)

                target_x = max(
                    0, min(target_x, screen_width - 1)
                )

                target_y = max(
                    0, min(target_y, screen_height - 1)
                )

                # ----------------------------
                # PINCH DETECTION
                # ----------------------------

                pinch_distance = math.hypot(
                    hand[4].x - hand[8].x,
                    hand[4].y - hand[8].y
                )

                palm_size = math.hypot(
                    hand[0].x - hand[9].x,
                    hand[0].y - hand[9].y
                )

                pinch_ratio = (
                    pinch_distance / max(palm_size, 0.001)
                )

                current_time = time.monotonic()

                # ----------------------------
                # GESTURE PRIORITY
                # ----------------------------

                index_extended = finger_extended(hand, 8, 6)
                middle_extended = finger_extended(hand, 12, 10)
                ring_extended = finger_extended(hand, 16, 14)
                pinky_extended = finger_extended(hand, 20, 18)

                open_palm = open_palm_detected(hand, palm_size)

                # Open palm takes priority over every mouse action.
                if open_palm:

                    pause_exit_counter = 0

                    if pause_gesture_start is None and not pause_gesture_latched:
                        pause_gesture_start = current_time

                    if paused:
                        gesture_text = "HOLD OPEN PALM TO RESUME"
                    else:
                        gesture_text = "HOLD OPEN PALM TO PAUSE"

                    gesture_colour = (0, 200, 255)

                    if (
                        not pause_gesture_latched
                        and pause_gesture_start is not None
                        and current_time - pause_gesture_start >= OPEN_PALM_HOLD_TIME
                    ):

                        paused = not paused
                        pause_gesture_latched = True
                        pause_gesture_start = None

                        gesture_text = "PAUSED" if paused else "RESUMED"
                        gesture_colour = (
                            (0, 0, 255) if paused else (0, 255, 0)
                        )

                    # Never leave a drag held while pausing.
                    if drag_active:
                        pyautogui.mouseUp()

                    pinch_tracking = False
                    drag_active = False
                    right_click_start_time = None
                    right_click_latched = False
                    right_click_exit_counter = 0

                    pinch_start_time = None
                    pinch_cursor_position = None
                    pinch_start_filtered_x = None
                    pinch_start_filtered_y = None

                    scroll_mode = False
                    scroll_exit_counter = 0
                    scroll_velocity = 0.0
                    scroll_accumulator = 0.0

                else:

                    # A fresh open-palm gesture is required for the next toggle.
                    if pause_gesture_latched:

                        pause_exit_counter += 1

                        if pause_exit_counter >= PAUSE_EXIT_GRACE_FRAMES:
                            pause_gesture_latched = False
                            pause_exit_counter = 0

                    else:
                        pause_gesture_start = None
                        pause_exit_counter = 0

                    if paused:

                        gesture_text = "PAUSED - OPEN PALM TO RESUME"
                        gesture_colour = (0, 0, 255)

                        if drag_active:
                            pyautogui.mouseUp()

                        pinch_tracking = False
                        drag_active = False
                        right_click_start_time = None
                        right_click_latched = False
                        right_click_exit_counter = 0

                        scroll_mode = False
                        scroll_exit_counter = 0
                        scroll_velocity = 0.0
                        scroll_accumulator = 0.0

                    else:

                        # ----------------------------
                        # RIGHT-CLICK GESTURE
                        # ----------------------------

                        # One raised middle finger; index, ring and pinky folded;
                        # thumb extended. Show the back of the hand to the camera.
                        right_click_gesture = (
                            middle_extended
                            and not index_extended
                            and not ring_extended
                            and not pinky_extended
                            and thumb_extended(hand, palm_size)
                        )

                        right_click_active = False
                        right_click_completed = False

                        if right_click_gesture:
                            right_click_exit_counter = 0

                            if right_click_start_time is None and not right_click_latched:
                                right_click_start_time = current_time

                            if right_click_latched:
                                right_click_active = True
                                gesture_text = "RIGHT CLICKED - RELEASE GESTURE"
                                gesture_colour = (0, 255, 0)

                            elif right_click_start_time is not None:
                                right_click_active = True
                                gesture_text = "HOLD FOR RIGHT CLICK"
                                gesture_colour = (255, 180, 0)

                                if current_time - right_click_start_time >= RIGHT_CLICK_HOLD_TIME:
                                    pyautogui.rightClick()
                                    right_click_latched = True
                                    right_click_start_time = None
                                    right_click_completed = True
                                    gesture_text = "RIGHT CLICKED"
                                    gesture_colour = (0, 255, 0)

                        elif right_click_latched or right_click_start_time is not None:
                            right_click_exit_counter += 1
                            right_click_active = True

                            if right_click_exit_counter > RIGHT_CLICK_EXIT_GRACE_FRAMES:
                                right_click_latched = False
                                right_click_start_time = None
                                right_click_exit_counter = 0
                                right_click_active = False

                        if right_click_active or right_click_completed:

                            # Cancel any in-progress left-click/drag and suppress
                            # scrolling until the right-click pose is released.
                            if drag_active:
                                pyautogui.mouseUp()

                            pinch_tracking = False
                            drag_active = False
                            pinch_start_time = None
                            pinch_cursor_position = None
                            pinch_start_filtered_x = None
                            pinch_start_filtered_y = None

                            scroll_mode = False
                            scroll_exit_counter = 0
                            scroll_velocity = 0.0
                            scroll_accumulator = 0.0

                        else:

                            # ----------------------------
                            # SCROLL GESTURE DETECTION
                            # ----------------------------

                            scroll_gesture = (
                                index_extended
                                and middle_extended
                                and not ring_extended
                                and not pinky_extended
                            )

                            if scroll_gesture:

                                scroll_exit_counter = 0
                                scroll_mode = True

                            elif scroll_mode:

                                scroll_exit_counter += 1

                                if scroll_exit_counter > SCROLL_EXIT_GRACE_FRAMES:
                                    scroll_mode = False
                                    scroll_exit_counter = 0

                            # ----------------------------
                            # SCROLL MODE
                            # ----------------------------

                            if scroll_mode:

                                if drag_active:
                                    pyautogui.mouseUp()

                                pinch_tracking = False
                                drag_active = False

                                pinch_start_time = None
                                pinch_cursor_position = None

                                pinch_start_filtered_x = None
                                pinch_start_filtered_y = None

                                # Average index and middle fingertip positions.
                                scroll_y = (
                                    hand[8].y + hand[12].y
                                ) / 2.0

                                # Update scroll velocity from hand position.
                                if scroll_gesture:
                                    scroll_velocity = calculate_scroll_speed(
                                        scroll_y
                                    )

                                gesture_text = "SCROLL MODE"
                                gesture_colour = (255, 200, 0)

                                # Draw large centred scroll zone.
                                cv2.rectangle(
                                    frame,
                                    (
                                        int(SCROLL_X_MIN * width),
                                        int(SCROLL_Y_MIN * height)
                                    ),
                                    (
                                        int(SCROLL_X_MAX * width),
                                        int(SCROLL_Y_MAX * height)
                                    ),
                                    (0, 165, 255),
                                    3
                                )

                                # Draw centre/dead-zone guide.
                                centre_y = int(SCROLL_CENTRE * height)

                                cv2.line(
                                    frame,
                                    (
                                        int(SCROLL_X_MIN * width),
                                        centre_y
                                    ),
                                    (
                                        int(SCROLL_X_MAX * width),
                                        centre_y
                                    ),
                                    (255, 255, 255),
                                    1
                                )

                                cv2.putText(
                                    frame,
                                    "SCROLL FASTER UP",
                                    (10, 105),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (255, 200, 0),
                                    2
                                )

                                cv2.putText(
                                    frame,
                                    "SCROLL FASTER DOWN",
                                    (10, height - 45),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (0, 165, 255),
                                    2
                                )

                            else:

                                # ----------------------------
                                # NORMAL CURSOR / PINCH / DRAG
                                # ----------------------------

                                # Draw normal movement region.
                                cv2.rectangle(
                                    frame,
                                    (int(X_MIN * width), int(Y_MIN * height)),
                                    (int(X_MAX * width), int(Y_MAX * height)),
                                    (255, 255, 0),
                                    2
                                )

                                if not pinch_tracking:

                                    if pinch_ratio < PINCH_START_THRESHOLD:

                                        pinch_tracking = True
                                        pinch_start_time = current_time

                                        pinch_cursor_position = pyautogui.position()

                                        pinch_start_filtered_x = filtered_x
                                        pinch_start_filtered_y = filtered_y

                                        gesture_text = "PINCHING..."
                                        gesture_colour = (0, 165, 255)

                                    else:

                                        pyautogui.moveTo(
                                            int(target_x),
                                            int(target_y)
                                        )

                                        gesture_text = "CURSOR ACTIVE"
                                        gesture_colour = (0, 255, 0)

                                else:

                                    if pinch_ratio <= PINCH_RELEASE_THRESHOLD:

                                        elapsed = (
                                            current_time - pinch_start_time
                                        )

                                        if not drag_active:

                                            if elapsed >= DRAG_HOLD_TIME:

                                                pyautogui.moveTo(
                                                    pinch_cursor_position.x,
                                                    pinch_cursor_position.y
                                                )

                                                pyautogui.mouseDown()

                                                drag_active = True

                                        if drag_active:

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

                                            drag_x = max(
                                                0,
                                                min(drag_x, screen_width - 1)
                                            )

                                            drag_y = max(
                                                0,
                                                min(drag_y, screen_height - 1)
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

                                        if drag_active:

                                            pyautogui.mouseUp()

                                            gesture_text = "DROPPED"
                                            gesture_colour = (0, 255, 0)

                                        else:

                                            pyautogui.click(
                                                x=pinch_cursor_position.x,
                                                y=pinch_cursor_position.y
                                            )

                                            gesture_text = "CLICKED"
                                            gesture_colour = (0, 255, 0)

                                        # Reset pinch state.
                                        pinch_tracking = False
                                        drag_active = False

                                        pinch_start_time = None
                                        pinch_cursor_position = None

                                        pinch_start_filtered_x = None
                                        pinch_start_filtered_y = None

                                        pyautogui.moveTo(
                                            int(target_x),
                                            int(target_y)
                                        )

                            # ----------------------------
                            # SCROLL OUTPUT
                            # ----------------------------

                            # Scroll only while the two-finger gesture is detected.
                            if scroll_gesture and scroll_mode:

                                scroll_velocity = calculate_scroll_speed(
                                    scroll_y
                                )

                                scroll_accumulator += scroll_velocity * dt

                                scroll_steps = int(scroll_accumulator)

                                if scroll_steps != 0:

                                    pyautogui.scroll(scroll_steps)

                                    scroll_accumulator -= scroll_steps

                            else:

                                scroll_velocity = 0.0
                                scroll_accumulator = 0.0

                # Keep the paused state visible even after the toggle frame.
                if paused:

                    cv2.putText(
                        frame,
                        "PAUSED",
                        (width - 130, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
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
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Scroll speed: {abs(scroll_velocity):.1f}",
                    (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

            else:

                # Release any held mouse button.
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

                right_click_start_time = None
                right_click_latched = False
                right_click_exit_counter = 0

                pause_gesture_start = None
                pause_gesture_latched = False
                pause_exit_counter = 0

                scroll_mode = False
                scroll_exit_counter = 0

                scroll_velocity = 0.0
                scroll_accumulator = 0.0

                # Keep the pause state until an open palm is shown again.
                cv2.putText(
                    frame,
                    "NO HAND DETECTED",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                if paused:
                    cv2.putText(
                        frame,
                        "PAUSED",
                        (width - 130, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

            # ----------------------------
            # FPS COUNTER
            # ----------------------------

            frame_duration = max(
                time.monotonic() - frame_start_time,
                0.001
            )

            instant_fps = 1.0 / frame_duration

            display_fps = (
                0.9 * display_fps + 0.1 * instant_fps
                if display_fps else instant_fps
            )

            cv2.putText(
                frame,
                f"FPS: {display_fps:.1f}",
                (10, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.imshow("Floating Cursor", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:

    # Safety: never leave the mouse button held down.
    pyautogui.mouseUp()

    camera.release()
    cv2.destroyAllWindows()