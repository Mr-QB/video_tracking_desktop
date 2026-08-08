import os
import shutil
import subprocess
from pathlib import Path

# Thư mục gốc của dataset hiện tại
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_TEST_DIR = os.path.join(SCRIPT_DIR, "dataset", "test")
ORIGINAL_DIR = os.path.join(DATASET_TEST_DIR, "original")

def copy_dir_contents(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copy_dir_contents(s, d)
        else:
            shutil.copy2(s, d)

def decode_sequence_bitstream(codec_dir_path, seq_name):
    bitstream_file = os.path.join(codec_dir_path, f"{seq_name}.mp4")
    
    if not os.path.exists(bitstream_file):
        # Có thể không phải file mp4 hoặc không tồn tại
        return
        
    dst_seq_path = os.path.join(codec_dir_path, seq_name)
    dst_img_dir = os.path.join(dst_seq_path, "img1")
    
    # Skip if already decoded
    if os.path.exists(dst_img_dir) and len(os.listdir(dst_img_dir)) > 0:
        print(f"[SKIP] Sequence {seq_name} already decoded in {os.path.basename(codec_dir_path)}.")
        return
        
    os.makedirs(dst_img_dir, exist_ok=True)
    
    print(f"Decoding {seq_name} in {os.path.basename(codec_dir_path)} from bitstream video...")
    
    # Decode video back to images
    cmd_decode = [
        "ffmpeg", "-y",
        "-i", bitstream_file,
        "-q:v", "2",
        f"{dst_img_dir}/%06d.jpg"
    ]
    subprocess.run(cmd_decode, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Copy seqinfo.ini from corresponding original sequence
    src_original_seq = os.path.join(ORIGINAL_DIR, seq_name)
    
    src_seqinfo = os.path.join(src_original_seq, "seqinfo.ini")
    dst_seqinfo = os.path.join(dst_seq_path, "seqinfo.ini")
    if os.path.exists(src_seqinfo):
        shutil.copy2(src_seqinfo, dst_seqinfo)
        
    # Copy ground truth (gt) if exists
    src_gt = os.path.join(src_original_seq, "gt")
    dst_gt = os.path.join(dst_seq_path, "gt")
    if os.path.exists(src_gt):
        copy_dir_contents(src_gt, dst_gt)
        
    # Copy detections (det) if exists
    src_det = os.path.join(src_original_seq, "det")
    dst_det = os.path.join(dst_seq_path, "det")
    if os.path.exists(src_det):
        copy_dir_contents(src_det, dst_det)
        
    print(f"✓ Decoded and populated folder for {seq_name} in {os.path.basename(codec_dir_path)}")

def main():
    print("==================================================")
    print("STARTING BITSTREAM DECODING PIPELINE")
    print("==================================================")
    
    if not os.path.exists(DATASET_TEST_DIR):
        print(f"Dataset directory not found: {DATASET_TEST_DIR}")
        return
        
    # Duyệt qua các thư mục trong dataset/test (trừ original)
    for codec_folder in os.listdir(DATASET_TEST_DIR):
        if codec_folder == "original":
            continue
            
        codec_dir_path = os.path.join(DATASET_TEST_DIR, codec_folder)
        if not os.path.isdir(codec_dir_path):
            continue
            
        print(f"\n>>> Processing folder: {codec_folder}")
        
        # Tìm các file mp4 trong thư mục codec (ví dụ QP22, QP27,...)
        for item in os.listdir(codec_dir_path):
            if item.endswith(".mp4"):
                seq_name = item[:-4] # bỏ .mp4
                decode_sequence_bitstream(codec_dir_path, seq_name)
        
    print("\n==================================================")
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    main()
