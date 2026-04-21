import cv2
import mediapipe as mp

# ---------------- MEDIAPIPE SETUP ---------------- #
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# ---------------- FUNCTIONS ---------------- #

def mouth_open_ratio(landmarks, w, h):
    # Mouth landmarks
    top = landmarks[13]
    bottom = landmarks[14]
    left = landmarks[78]
    right = landmarks[308]

    # Convert to pixel coords
    top = (int(top.x * w), int(top.y * h))
    bottom = (int(bottom.x * w), int(bottom.y * h))
    left = (int(left.x * w), int(left.y * h))
    right = (int(right.x * w), int(right.y * h))

    vertical = abs(top[1] - bottom[1])
    horizontal = abs(left[0] - right[0])

    ratio = vertical / (horizontal + 1e-6)

    return ratio, top, bottom, left, right


def head_tilt_distance(landmarks, w, h):
    # Nose and chin landmarks
    nose = landmarks[1]
    chin = landmarks[152]

    nose = (int(nose.x * w), int(nose.y * h))
    chin = (int(chin.x * w), int(chin.y * h))

    distance = abs(chin[1] - nose[1])

    return distance, nose, chin


# ---------------- CONFIG ---------------- #
YAWN_THRESHOLD = 0.6
YAWN_FRAMES = 15

TILT_THRESHOLD = 120
TILT_FRAMES = 15

yawn_counter = 0
tilt_counter = 0

# ---------------- CAMERA ---------------- #
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ---------------- MAIN LOOP ---------------- #
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
) as face_mesh:

    cv2.namedWindow("DRIVER MONITORING SYSTEM", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("DRIVER MONITORING SYSTEM", cv2.WND_PROP_TOPMOST, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror frame
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Title
        cv2.putText(frame, "DROWSINESS DETECTION SYSTEM",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2)

        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        detected = results.multi_face_landmarks is not None

        # Detection status
        cv2.putText(frame, f"Face Detected: {detected}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0) if detected else (0, 0, 255),
                    2)

        if detected:
            for lm in results.multi_face_landmarks:

                # -------- YAWNING DETECTION -------- #
                mar, top, bottom, left, right = mouth_open_ratio(lm.landmark, w, h)

                # Draw mouth
                cv2.circle(frame, top, 3, (255, 0, 0), -1)
                cv2.circle(frame, bottom, 3, (255, 0, 0), -1)
                cv2.circle(frame, left, 3, (255, 0, 0), -1)
                cv2.circle(frame, right, 3, (255, 0, 0), -1)
                cv2.line(frame, top, bottom, (0, 255, 255), 2)

                # Yawn logic
                if mar > YAWN_THRESHOLD:
                    yawn_counter += 1
                    if yawn_counter > YAWN_FRAMES:
                        cv2.putText(frame, "YAWNING DETECTED!",
                                    (20, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9,
                                    (0, 0, 255),
                                    3)
                else:
                    yawn_counter = 0

                # -------- HEAD TILT DETECTION -------- #
                tilt_dist, nose, chin = head_tilt_distance(lm.landmark, w, h)

                # Draw tilt line
                cv2.line(frame, nose, chin, (255, 255, 0), 2)

                # Tilt logic
                if tilt_dist < TILT_THRESHOLD:
                    tilt_counter += 1
                    if tilt_counter > TILT_FRAMES:
                         cv2.putText(frame, "HEAD DOWN!",
                                    (20, 170),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9,
                                    (0, 0, 255),
                                    3)
                else:
                    tilt_counter = 0

                # -------- COMBINED ALERT -------- #
                if (yawn_counter > YAWN_FRAMES) or (tilt_counter > TILT_FRAMES):
                    cv2.putText(frame, "DROWSINESS ALERT!",
                                (20, 230),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255),
                                3)

  
                # -------- DEBUG INFO -------- #
                cv2.putText(frame, f"MAR: {mar:.2f}",
  
                            (20, 270),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2)

                cv2.putText(frame, f"Tilt: {tilt_dist}",
                            (20, 300),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2)

                # Draw mesh
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=lm,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=1, circle_radius=1
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 0, 0), thickness=1
                    ),
                )

        cv2.imshow("DRIVER MONITORING SYSTEM", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ---------------- CLEANUP ---------------- #
cap.release()
cv2.destroyAllWindows()