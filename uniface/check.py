from ultralytics import YOLO

model = YOLO("/home/vivek/Downloads/yolo26s-face.pt")

print("Task:", model.task)
print("Classes:", model.names)

for result in model(
    "/home/vivek/Music/face/employee/20000_vivek.jpg",
    imgsz=640,
    conf=0.5,
    device=0
):
    print("Keypoints:", result.keypoints)

    if result.keypoints is not None:
        print("Keypoint coordinates:")
        print(result.keypoints.xy.cpu().numpy())