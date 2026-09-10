import os
import json
import sys
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from core.config import get_path

# Patch numpy aliases for TrackEval compatibility
np.float = float
np.int = int
np.bool = bool

# Add official internal Jonathon Luiten TrackEval repository to sys.path
trackeval_path = str(Path(__file__).parent.parent / "TrackEval")
if trackeval_path not in sys.path:
    sys.path.insert(0, trackeval_path)

import trackeval

def calculate_official_trackeval_metrics(gt_file, track_file_or_df):
    """
    Evaluates tracking performance using Jonathon Luiten's official TrackEval package:
    https://github.com/JonathonLuiten/TrackEval
    Returns: HOTA, IDF1, MOTA, ID_Switches, False_Negatives, Recall, Precision
    """
    if not os.path.exists(gt_file):
        return None

    try:
        df_gt = pd.read_csv(gt_file, header=None)
        
        if isinstance(track_file_or_df, (str, Path)):
            if not os.path.exists(track_file_or_df):
                return None
            df_ts = pd.read_csv(track_file_or_df, header=None)
        elif isinstance(track_file_or_df, pd.DataFrame):
            df_ts = track_file_or_df
        else:
            return None
        
        if df_gt.empty or df_ts.empty:
            return None

        # Filter active groundtruth targets
        if df_gt.shape[1] >= 7:
            df_gt = df_gt[df_gt[6] != 0]
            
        unique_gt_map = {gt_id: i for i, gt_id in enumerate(sorted(df_gt[1].unique()))}
        unique_tr_map = {tr_id: i for i, tr_id in enumerate(sorted(df_ts[1].unique()))}
        
        frames = sorted(list(set(df_gt[0].unique()).union(set(df_ts[0].unique()))))
        gt_ids, tracker_ids, similarity_scores = [], [], []
        gt_dets, tracker_dets = 0, 0
        
        for f in frames:
            gt_f = df_gt[df_gt[0] == f]
            ts_f = df_ts[df_ts[0] == f]
            
            g_ids = np.array([unique_gt_map[x] for x in gt_f[1].values], dtype=int) if not gt_f.empty else np.empty(0, dtype=int)
            t_ids = np.array([unique_tr_map[x] for x in ts_f[1].values], dtype=int) if not ts_f.empty else np.empty(0, dtype=int)
            
            g_boxes = gt_f[[2, 3, 4, 5]].values if not gt_f.empty else np.empty((0, 4))
            t_boxes = ts_f[[2, 3, 4, 5]].values if not ts_f.empty else np.empty((0, 4))
            
            gt_ids.append(g_ids)
            tracker_ids.append(t_ids)
            gt_dets += len(g_ids)
            tracker_dets += len(t_ids)
            
            if len(g_boxes) > 0 and len(t_boxes) > 0:
                iou_mat = trackeval.datasets._base_dataset._BaseDataset._calculate_box_ious(g_boxes, t_boxes, box_format='xywh')
            else:
                iou_mat = np.empty((len(g_boxes), len(t_boxes)))
            similarity_scores.append(iou_mat)
            
        raw_data = {
            'num_timesteps': len(frames),
            'num_gt_dets': gt_dets,
            'num_tracker_dets': tracker_dets,
            'gt_ids': gt_ids,
            'tracker_ids': tracker_ids,
            'similarity_scores': similarity_scores,
            'num_gt_ids': len(unique_gt_map),
            'num_tracker_ids': len(unique_tr_map)
        }
        
        hota_metric = trackeval.metrics.HOTA({'PRINT_CONFIG': False})
        clear_metric = trackeval.metrics.CLEAR({'PRINT_CONFIG': False})
        identity_metric = trackeval.metrics.Identity({'PRINT_CONFIG': False})
        
        hota_res = hota_metric.eval_sequence(raw_data)
        clear_res = clear_metric.eval_sequence(raw_data)
        id_res = identity_metric.eval_sequence(raw_data)
        
        hota = float(np.mean(hota_res['HOTA'])) * 100
        idf1 = float(np.mean(id_res['IDF1'])) * 100
        mota = float(np.mean(clear_res['MOTA'])) * 100
        switches = int(np.sum(clear_res['IDSW']))
        misses = int(np.sum(clear_res['CLR_FN']))
        recall = float(np.mean(clear_res['CLR_Re'])) * 100
        precision = float(np.mean(clear_res['CLR_Pr'])) * 100
        
        return {
            'HOTA': round(max(0.0, hota), 1),
            'IDF1': round(max(0.0, idf1), 1),
            'MOTA': round(max(0.0, mota), 1),
            'ID_Switches': max(0, switches),
            'False_Negatives': max(0, misses),
            'Recall': round(max(0.0, recall), 1),
            'Precision': round(max(0.0, precision), 1)
        }
    except Exception as e:
        print(f"[WARN] Error in Jonathon Luiten TrackEval: {e}")
        return None

def evaluate_and_cache_metrics(method_dir, method_name, is_baseline=False, seq_name="MOT20-01", codec_name="QP37"):
    dataset_dir = get_path("dataset_images_dir", "dataset/test")
    gt_file = dataset_dir / "original" / seq_name / "gt" / "gt.txt"
    
    target_track_filename = "comp_tracks.txt" if is_baseline else "enh_tracks.txt"
    
    # 1. Check for metrics.json in method_dir or subfolder
    json_candidates = [
        os.path.join(method_dir, "metrics.json"),
        os.path.join(method_dir, seq_name, "metrics.json"),
    ]
    for j_path in json_candidates:
        if os.path.exists(j_path):
            try:
                with open(j_path, 'r') as f:
                    data = json.load(f)
                    if "HOTA" in data and "IDF1" in data:
                        return data
            except Exception:
                pass

    # 2. Candidate track files
    track_candidates = [
        os.path.join(method_dir, target_track_filename),
        os.path.join(method_dir, seq_name, target_track_filename)
    ]

    track_file = next((p for p in track_candidates if os.path.exists(p)), None)
    
    if track_file and os.path.exists(gt_file):
        official_metrics = calculate_official_trackeval_metrics(str(gt_file), str(track_file))
        if official_metrics:
            try:
                os.makedirs(method_dir, exist_ok=True)
                with open(os.path.join(method_dir, "metrics.json"), 'w') as f:
                    json.dump(official_metrics, f, indent=4)
            except Exception:
                pass
            return official_metrics
            
    print(f"[WARN] No verified tracking metrics available for {method_name} ({seq_name}).")
    return None
