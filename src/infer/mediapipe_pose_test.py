import cv2
import mediapipe as mp
#Webcam skeleton detection verified
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not accessible")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as pose:

        print("MediaPipe Pose running. Press 'q' to exit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            cv2.imshow("MediaPipe Pose — Full Body + Feet + Toes", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
