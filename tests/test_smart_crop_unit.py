
import os
import pytest
from moviepy.video.io.VideoFileClip import VideoFileClip
from src.worker.services.smart_crop.detectors.face_detector import FaceDetector
from src.worker.services.smart_crop.detectors.text_detector import TextDetector
from src.worker.services.smart_crop.smart_crop_engine import SmartCropEngine
from src.worker.services.smart_crop_service import render_dynamic_crop

# Path to the existing video downloaded by worker
TEST_VIDEO_PATH = "downloads/1/58/video.mp4"
OUTPUT_PATH = "tests/test_smart_crop_output.mp4"

@pytest.mark.skipif(not os.path.exists(TEST_VIDEO_PATH), reason="Test video not found")
def test_smart_crop_pipeline():
    print(f"Testing Smart Crop on snippet of {TEST_VIDEO_PATH}")
    
    # 1. Create a short clip (3 seconds)
    original_clip = VideoFileClip(TEST_VIDEO_PATH)
    # Use a segment likely to have content (e.g. 10s to 13s)
    start_t = 10
    end_t = 13
    if original_clip.duration < 13:
        start_t = 0
        end_t = min(3, original_clip.duration)
        
    clip = original_clip.subclipped(start_t, end_t)
    
    # 2. Initialize Engine
    detectors = [
        FaceDetector(),
        TextDetector(backend="east")
    ]
    engine = SmartCropEngine(detectors, frame_sample_interval=0.5)
    
    # 3. Process
    crop_boxes = engine.process_video(clip)
    
    assert len(crop_boxes) > 0, "Engine should return crop boxes"
    
    # Check structure of crop_boxes
    first_box = crop_boxes[0]
    assert len(first_box) == 2
    assert isinstance(first_box[0], float) # timestamp
    assert len(first_box[1]) == 4 # (x1, y1, x2, y2)
    
    # 4. Render
    target_w = 1080
    target_h = 1920
    
    output_file = render_dynamic_crop(clip, crop_boxes, target_w, target_h, OUTPUT_PATH)
    
    assert os.path.exists(output_file)
    
    # 5. Verify Output Dimensions
    result_clip = VideoFileClip(output_file)
    assert tuple(result_clip.size) == (target_w, target_h)
    assert abs(result_clip.duration - clip.duration) < 0.1
    
    result_clip.close()
    clip.close()
    original_clip.close()
    
    # Cleanup
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
