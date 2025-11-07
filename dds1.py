import time
import math
import cv2
import numpy as np
import mediapipe as mp
import winsound 
LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_POINTS = [263, 387, 385, 362, 380, 373]


MOUTH_POINTS = [61, 81, 13, 311, 308, 402, 14, 178]


def euclidean_distance(p1, p2):
    return float(np.linalg.norm(p1 - p2))


def compute_ear(landmarks_2d, points):
   
    p1, p2, p3, p4, p5, p6 = [landmarks_2d[idx] for idx in points]
    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def compute_mar(landmarks_2d, points):
    p1, p2, p3, p4, p5, p6, p7, p8 = [landmarks_2d[idx] for idx in points]
    vertical_1 = euclidean_distance(p2, p8)
    vertical_2 = euclidean_distance(p3, p7)
    vertical_3 = euclidean_distance(p4, p6)
    horizontal = euclidean_distance(p1, p5)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2 + vertical_3) / (3.0 * horizontal)



def play_alert_sound():

    frequency = 3000  
    duration = 2000   
    winsound.Beep(frequency, duration)
    print(" Alert Sound Played!")


def run_drowsiness_detection():

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    EAR_THRESHOLD = 0.25
    MAR_THRESHOLD = 0.65 
    CONSECUTIVE_FRAMES = 30
    YAWN_FRAMES = 10
    COOLDOWN_SECONDS = 2.0

    consecutive_below = 0
    yawn_counter = 0
    last_alert_time = 0.0

    print("Drowsiness + Yawning Detection Started!")
    print("Press 'q' or ESC to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(frame_rgb)

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0]
                h, w = frame.shape[:2]
                landmarks_2d = np.array([[int(l.x * w), int(l.y * h)] for l in landmarks.landmark])

                
                left_ear = compute_ear(landmarks_2d, LEFT_EYE_POINTS)
                right_ear = compute_ear(landmarks_2d, RIGHT_EYE_POINTS)
                mouth_mar = compute_mar(landmarks_2d, MOUTH_POINTS)

                if left_ear > 0 and right_ear > 0:
                    ear = (left_ear + right_ear) / 2.0

                    
                    cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"MAR: {mouth_mar:.3f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    
                    if ear < EAR_THRESHOLD:
                        consecutive_below += 1
                        if consecutive_below >= CONSECUTIVE_FRAMES:
                            current_time = time.time()
                            if current_time - last_alert_time > COOLDOWN_SECONDS:
                                cv2.putText(frame, "DROWSINESS DETECTED!", (10, 120),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                                play_alert_sound()
                                last_alert_time = current_time
                    else:
                        consecutive_below = 0

                    
                    if mouth_mar > MAR_THRESHOLD:
                        yawn_counter += 1
                        if yawn_counter >= YAWN_FRAMES:
                            current_time = time.time()
                            if current_time - last_alert_time > COOLDOWN_SECONDS:
                                cv2.putText(frame, "YAWNING DETECTED!", (10, 160),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3)
                                play_alert_sound()
                                last_alert_time = current_time
                    else:
                        yawn_counter = 0

                    
                    if ear >= EAR_THRESHOLD and mouth_mar <= MAR_THRESHOLD:
                        cv2.putText(frame, "AWAKE", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    consecutive_below = 0
                    yawn_counter = 0

            cv2.imshow("Drowsiness + Yawning Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()


if __name__ == "__main__":
    run_drowsiness_detection()
