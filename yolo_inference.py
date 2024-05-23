from ultralytics import YOLO
model = YOLO('yolov8l')


results = model.predict('input_videos/input.mp4', save= True)
print(results[0])
print("+++++++++++++++++++++++++++++++++++++++++=")
for box in results.boxes:
    print(box)