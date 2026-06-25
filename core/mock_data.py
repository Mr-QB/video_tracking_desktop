import numpy as np

TOTAL_FRAMES = 1500
OBJECT_IDS = [3, 7, 12, 15, 23]

# Define key events where tracking usually fails
EVENTS = {
    120: {"name": "Blur", "type": "motion"},
    280: {"name": "Occlusion", "type": "crowd"},
    510: {"name": "Lost Target", "type": "occlusion"},
    700: {"name": "ID Switch", "type": "compressed"},
    940: {"name": "Recovery", "type": "re-acquired"},
    1220: {"name": "Occlusion", "type": "vehicle"}
}

def generate_timeline_data():
    """Generates synthetic IDF1 data for both compressed and enhanced videos."""
    np.random.seed(42)
    frames = np.arange(TOTAL_FRAMES)
    
    # Base IDF1
    compressed_idf1 = np.random.normal(45, 3, TOTAL_FRAMES)
    enhanced_idf1 = np.random.normal(75, 2, TOTAL_FRAMES)
    
    # Smooth them a bit
    compressed_idf1 = np.convolve(compressed_idf1, np.ones(10)/10, mode='same')
    enhanced_idf1 = np.convolve(enhanced_idf1, np.ones(10)/10, mode='same')
    
    # Apply events
    for frame, event in EVENTS.items():
        # Compressed drops significantly
        drop_comp = np.random.uniform(15, 25)
        length = np.random.randint(50, 100)
        start = max(0, frame - 10)
        end = min(TOTAL_FRAMES, frame + length)
        compressed_idf1[start:end] -= drop_comp * np.exp(-np.linspace(0, 3, end-start))
        
        # Enhanced drops slightly
        drop_enh = np.random.uniform(2, 8)
        length_enh = np.random.randint(20, 50)
        end_enh = min(TOTAL_FRAMES, frame + length_enh)
        enhanced_idf1[start:end_enh] -= drop_enh * np.exp(-np.linspace(0, 3, end_enh-start))
        
    # Clip to valid range
    compressed_idf1 = np.clip(compressed_idf1, 0, 100)
    enhanced_idf1 = np.clip(enhanced_idf1, 0, 100)
    
    return frames, compressed_idf1, enhanced_idf1

def get_failure_cases():
    """Returns mock data for the failure analysis table."""
    return [
        {"frame": 420, "id": 7, "type": "Car", "comp_res": "Lost track (occlusion by bus)", "enh_res": "Track maintained through occlusion", "status": "Improved", "notes": "Recovered at frame 512"},
        {"frame": 690, "id": 12, "type": "Ped", "comp_res": "ID Switch (12 -> 28)", "enh_res": "Stable ID", "status": "Improved", "notes": "Compressed ID switch at frame 689"},
        {"frame": 1050, "id": 3, "type": "Ped", "comp_res": "Stable ID", "enh_res": "Stable ID", "status": "Neutral", "notes": "Consistent in both"},
        {"frame": 1230, "id": 23, "type": "Car", "comp_res": "Wrong ID (23 -> 31)", "enh_res": "Correct ID (23)", "status": "Improved", "notes": "Appearance confusion in compressed"},
        {"frame": 1440, "id": 15, "type": "Car", "comp_res": "Lost track", "enh_res": "Track maintained", "status": "Improved", "notes": "Low resolution caused miss"}
    ]

def get_tracking_data_for_frame(frame_idx, is_enhanced):
    """
    Generates mock bounding boxes for a specific frame.
    Returns a list of dicts: [{'id': int, 'bbox': [x, y, w, h], 'conf': float}]
    """
    np.random.seed(frame_idx)
    objects = []
    
    for obj_id in OBJECT_IDS:
        # Simulate movement
        x = 200 + (obj_id * 50) + (frame_idx % 500) * 0.5 + np.random.normal(0, 5)
        y = 300 + np.sin(frame_idx / 50.0 + obj_id) * 50 + np.random.normal(0, 5)
        w = 50 + (obj_id % 3) * 10
        h = 100 + (obj_id % 2) * 20
        
        conf = np.random.uniform(0.7, 0.99) if is_enhanced else np.random.uniform(0.4, 0.85)
        
        current_id = obj_id
        
        # Simulate tracking errors in compressed video
        if not is_enhanced:
            # Drop some tracks
            if obj_id == 7 and 420 <= frame_idx <= 512:
                continue
            if obj_id == 15 and frame_idx >= 1440:
                continue
            
            # ID Switches
            if obj_id == 12 and frame_idx >= 690:
                current_id = 28
            if obj_id == 23 and frame_idx >= 1230:
                current_id = 31
                
            # Random dropouts due to low confidence
            if np.random.rand() > 0.95:
                continue
                
            # Wobbly bounding boxes
            x += np.random.normal(0, 10)
            y += np.random.normal(0, 10)
            w += np.random.normal(0, 5)
            h += np.random.normal(0, 5)
            
        objects.append({
            'id': current_id,
            'bbox': [int(x), int(y), int(w), int(h)],
            'conf': round(conf, 2)
        })
        
    return objects
