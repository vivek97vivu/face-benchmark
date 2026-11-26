import cv2
import time

# ============================================================
# CONFIG
# ============================================================

CAMERA_ID = 4

MODEL_PATH = (
    "/home/vivek/Music/face/yuNet + MobileFaceNe/face_detection_yunet_2023mar.onnx"
)

CONFIDENCE_THRESHOLD = 0.4


# ============================================================
# LOAD YUNET
# ============================================================

print("Loading YuNet...")

detector = cv2.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    (320, 320),
    CONFIDENCE_THRESHOLD,
    0.3,
    5000
)

print("YuNet loaded.")


# ============================================================
# CAMERA
# ============================================================

print(f"Opening camera /dev/video{CAMERA_ID}...")

cap = cv2.VideoCapture(
    CAMERA_ID,
    cv2.CAP_V4L2
)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise RuntimeError(
        f"Cannot open camera {CAMERA_ID}"
    )

print("Camera started.")
print("Press Q to quit.")


# ============================================================
# FPS
# ============================================================

prev_time = time.time()


# ============================================================
# LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera frame")
        break

    h, w = frame.shape[:2]

    # YuNet needs the actual frame size
    detector.setInputSize((w, h))

    # --------------------------------------------------------
    # Detect faces
    # --------------------------------------------------------

    _, faces = detector.detect(frame)

    face_count = 0

    if faces is not None:

        face_count = len(faces)

        for face in faces:

            # ------------------------------------------------
            # YuNet output:
            #
            # x, y, width, height,
            # right_eye,
            # left_eye,
            # nose,
            # right_mouth,
            # left_mouth,
            # confidence
            # ------------------------------------------------

            x = int(face[0])
            y = int(face[1])
            fw = int(face[2])
            fh = int(face[3])

            confidence = float(face[-1])

            # Keep inside frame
            x = max(0, x)
            y = max(0, y)

            x2 = min(w, x + fw)
            y2 = min(h, y + fh)

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Face {confidence:.2f}",
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # Landmarks
            # ------------------------------------------------

            # right eye
            cv2.circle(
                frame,
                (int(face[4]), int(face[5])),
                3,
                (255, 0, 0),
                -1
            )

            # left eye
            cv2.circle(
                frame,
                (int(face[6]), int(face[7])),
                3,
                (255, 0, 0),
                -1
            )

            # nose
            cv2.circle(
                frame,
                (int(face[8]), int(face[9])),
                3,
                (255, 0, 0),
                -1
            )

            # right mouth
            cv2.circle(
                frame,
                (int(face[10]), int(face[11])),
                3,
                (255, 0, 0),
                -1
            )

            # left mouth
            cv2.circle(
                frame,
                (int(face[12]), int(face[13])),
                3,
                (255, 0, 0),
                -1
            )

    # ========================================================
    # FPS
    # ========================================================

    now = time.time()

    fps = 1.0 / max(
        now - prev_time,
        1e-6
    )

    prev_time = now

    # ========================================================
    # INFO
    # ========================================================

    cv2.putText(
        frame,
        f"Faces: {face_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "YuNet | Camera 4 | Press Q to quit",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "YuNet Face Detection",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("Camera stopped.")
