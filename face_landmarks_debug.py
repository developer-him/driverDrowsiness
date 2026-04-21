import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

LANDMARK_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2)
CONNECTION_STYLE = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
) as face_mesh:

    cv2.namedWindow("DEBUG FACE MESH", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("DEBUG FACE MESH", cv2.WND_PROP_TOPMOST, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror view
        frame = cv2.flip(frame, 1)

        # ALWAYS draw a red bar so we know this file is running
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 255), -1)
        cv2.putText(frame, "DEBUG MODE", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        detected = results.multi_face_landmarks is not None

        # Show detected flag
        cv2.putText(
            frame,
            f"Detected: {detected}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if detected else (0, 0, 255),
            2,
        )

        if detected:
            # draw only contours to keep it clear
            for lm in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=lm,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=LANDMARK_STYLE,
                    connection_drawing_spec=CONNECTION_STYLE,
                )

                # also draw ONE big dot on the nose (index 1)
                nose = lm.landmark[1]
                h, w, _ = frame.shape
                x, y = int(nose.x * w), int(nose.y * h)
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

        cv2.imshow("DEBUG FACE MESH", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
