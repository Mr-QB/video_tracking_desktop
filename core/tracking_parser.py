import os

def parse_mot_tracking_file(filepath, frame_offset=0):
    """
    Parses a MOT format tracking file.
    Expected format (comma separated):
    <frame_idx>, <track_id>, <bbox_left>, <bbox_top>, <bbox_width>, <bbox_height>, <confidence>, <class_id>
    
    Returns:
        A dictionary mapping frame_idx (0-indexed) to a list of track dictionaries:
        {
            0: [
                {'id': 1, 'bbox': [100, 200, 50, 80], 'conf': 0.95, 'class': 0},
                ...
            ],
            ...
        }
    """
    tracks_by_frame = {}
    
    if not os.path.exists(filepath):
        print(f"[WARN] Tracking file not found: {filepath}")
        return tracks_by_frame
        
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 8:
                continue
                
            try:
                frame_idx = int(parts[0]) + frame_offset
                track_id = int(parts[1])
                x = int(float(parts[2]))
                y = int(float(parts[3]))
                w = int(float(parts[4]))
                h = int(float(parts[5]))
                conf = float(parts[6])
                class_id = int(float(parts[7]))
                
                if frame_idx not in tracks_by_frame:
                    tracks_by_frame[frame_idx] = []
                    
                tracks_by_frame[frame_idx].append({
                    'id': track_id,
                    'bbox': [x, y, w, h],
                    'conf': conf,
                    'class': class_id
                })
            except ValueError as e:
                print(f"[WARN] Failed to parse tracking line: {line} - {e}")
                continue
                
    return tracks_by_frame
