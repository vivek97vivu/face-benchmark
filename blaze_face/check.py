import cv2
import numpy as np
import time
from ai_edge_litert.interpreter import Interpreter


# ============================================================
# CONFIG
# ============================================================

CAMERA_ID = 4

MODEL_PATH = "/home/vivek/Music/face/blaze_face/blaze_face_short_range.tflite"

INPUT_SIZE = 128

CONFIDENCE_THRESHOLD = 0.2


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading BlazeFace model...")

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nInput:")
print(input_details)

print("\nOutputs:")
for out in output_details:
    print(out)


input_index = input_details[0]["index"]
input_shape = input_details[0]["shape"]
input_dtype = input_details[0]["dtype"]

print("\nInput shape :", input_shape)
print("Input dtype :", input_dtype)


# ============================================================
# CAMERA
# ============================================================

print(f"\nOpening camera /dev/video{CAMERA_ID}...")

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera {CAMERA_ID}"
    )

print("Camera started.")
print("Press Q to quit.")


# ============================================================
# FPS
# ============================================================

prev_time = time.time()
fps = 0


# ============================================================
# INFERENCE
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    frame_h, frame_w = frame.shape[:2]

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    img = cv2.resize(
        frame,
        (INPUT_SIZE, INPUT_SIZE)
    )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Prepare tensor
    # --------------------------------------------------------

    if input_dtype == np.float32:

        img_input = img.astype(np.float32)

        # BlazeFace models commonly expect [0,1]
        img_input = img_input / 255.0

    elif input_dtype == np.uint8:

        img_input = img.astype(np.uint8)

    elif input_dtype == np.int8:

        img_input = img.astype(np.int8)

    else:

        raise RuntimeError(
            f"Unsupported input dtype: {input_dtype}"
        )

    img_input = np.expand_dims(
        img_input,
        axis=0
    )

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    interpreter.set_tensor(
        input_index,
        img_input
    )

    interpreter.invoke()

    outputs = []

    for out in output_details:

        outputs.append(
            interpreter.get_tensor(
                out["index"]
            )
        )

    # --------------------------------------------------------
    # Print output shapes once
    # --------------------------------------------------------

    if not hasattr(
        globals(),
        "_printed_outputs"
    ):

        print("\nOutput shapes:")

        for i, output in enumerate(outputs):
            print(
                i,
                output.shape,
                output.dtype
            )

        _printed_outputs = True

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Camera: {CAMERA_ID}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    current_time = time.time()

    fps = 1.0 / max(
        current_time - prev_time,
        1e-6
    )

    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "BlazeFace Camera Test",
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
