from ultralytics import YOLO
import supervision as sv
class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model=model_path)
        self.tracker = sv.ByteTrack()
    
    def detect_frames(self, frames):
        detections = [] 
        batch_size = 20 # 20 frames will be detected at a time
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf = 0.1)
            detections += detections_batch
            break ##JUST FOR NOW
        return detections

    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)

        for frame_num, detection in enumerate(detections):
            class_names = detection.names # 0:player 1:goalkeeper 2:ball
            class_names_inversed = {v:k for k,v in class_names.items()} # player:0 goalkeeper:1 ball:2

            #converting into supervision detection format
            supervision_detection = sv.Detections.from_ultralytics(detection)

            print(supervision_detection)