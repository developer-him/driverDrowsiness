import cv2
import mediapipe as mp
import numpy as np
import winsound
import os

# === CONFIG ===
ALARM_FILE = "alarm.wav"   # your custom alarm sound in the same folder
EAR_THRESHOLD = 0.22       # below this => eyes considered closed
CLOSED_FRAMES_THRESHOLD = 15  # frames with low EAR => drowsy
MAR_THRESHOLD = 0.5        # above this => mouth considered wide open (yawn-ish)
YAWN_FRAMES_THRESHOLD = 15  # frames with high MAR => yawn

# Eye landmark indices from MediaPipe FaceMesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Mouth landmark indices (corners + upper/lower lip center)
MOUTH_LEFT = 78
MOUTH_RIGHT = 308
MOUTH_TOP = 13
MOUTH_BOTTOM = 14


def compute_ear(landmarks, eye_indices, w, h):
    """Eye Aspect Ratio calculation for one eye."""
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append(np.array([lm.x * w, lm.y * h]))

    p1, p2, p3, p4, p5, p6 = pts

    # vertical distances
    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    # horizontal distance
    C = np.linalg.norm(p1 - p4)

    if C == 0:
        return 0.0

    return (A + B) / (2.0 * C)


def compute_mar(landmarks, w, h):
    """Mouth Aspect Ratio: how open the mouth is."""
    top = landmarks[MOUTH_TOP]
    bottom = landmarks[MOUTH_BOTTOM]
    left = landmarks[MOUTH_LEFT]
    right = landmarks[MOUTH_RIGHT]

    p_top = np.array([top.x * w, top.y * h])
    p_bottom = np.array([bottom.x * w, bottom.y * h])
    p_left = np.array([left.x * w, left.y * h])
    p_right = np.array([right.x * w, right.y * h])

    vertical = np.linalg.norm(p_top - p_bottom)
    horizontal = np.linalg.norm(p_left - p_right)

    if horizontal == 0:
        return 0.0

    return vertical / horizontal


mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Counters
closed_counter = 0
yawn_counter = 0

alarm_on = False

# Resolve alarm file path
alarm_path = os.path.join(os.path.dirname(__file__), ALARM_FILE)
alarm_available = os.path.exists(alarm_path)
print(f"Custom alarm file found: {alarm_available} ({alarm_path})")

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    cv2.namedWindow("DROWSINESS + YAWN DETECTOR", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("DROWSINESS + YAWN DETECTOR", cv2.WND_PROP_TOPMOST, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        status = "NO FACE"
        color = (0, 0, 255)
        ear_value = None
        mar_value = None

        is_drowsy_eyes = False
        is_yawning = False

        if results.multi_face_landmarks:
            face_lm = results.multi_face_landmarks[0].landmark

            # --- EAR (eyes) ---
            left_ear = compute_ear(face_lm, LEFT_EYE, w, h)
            right_ear = compute_ear(face_lm, RIGHT_EYE, w, h)
            ear_value = (left_ear + right_ear) / 2.0

            if ear_value < EAR_THRESHOLD:
                closed_counter += 1
            else:
                closed_counter = 0

            if closed_counter >= CLOSED_FRAMES_THRESHOLD:
                is_drowsy_eyes = True

            # --- MAR (mouth / yawn) ---
            mar_value = compute_mar(face_lm, w, h)

            if mar_value > MAR_THRESHOLD:
                yawn_counter += 1
            else:
                yawn_counter = 0

            if yawn_counter >= YAWN_FRAMES_THRESHOLD:
                is_yawning = True

            # --- STATUS TEXT LOGIC ---
            if is_drowsy_eyes and is_yawning:
                status = "DROWSY + YAWNING!"
            elif is_drowsy_eyes:
                status = "DROWSY!"
            elif is_yawning:
                status = "YAWNING!"
            else:
                status = "ALERT"

            color = (0, 255, 0) if status == "ALERT" else (0, 0, 255)

            # --- ALARM: trigger if eyes drowsy OR yawning long ---
            should_alarm = is_drowsy_eyes or is_yawning

            if should_alarm and not alarm_on:
                if alarm_available:
                    winsound.PlaySound(
                        alarm_path,
                        winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP
                    )
                else:
                    winsound.PlaySound(
                        "SystemHand",
                        winsound.SND_ALIAS | winsound.SND_ASYNC | winsound.SND_LOOP
                    )
                alarm_on = True
            elif not should_alarm and alarm_on:
                winsound.PlaySound(None, winsound.SND_PURGE)
                alarm_on = False

            # Draw eye and mouth landmarks
            for idx in LEFT_EYE + RIGHT_EYE + [MOUTH_LEFT, MOUTH_RIGHT, MOUTH_TOP, MOUTH_BOTTOM]:
                lm = face_lm[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)

        # Display EAR and MAR
        if ear_value is not None:
            cv2.putText(frame, f"EAR: {ear_value:.3f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if mar_value is not None:
            cv2.putText(frame, f"MAR: {mar_value:.3f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Status text
        cv2.putText(frame, status, (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        cv2.imshow("DROWSINESS + YAWN DETECTOR", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# stop alarm if still playing
if alarm_on:
    winsound.PlaySound(None, winsound.SND_PURGE)

cap.release()
cv2.destroyAllWindows()