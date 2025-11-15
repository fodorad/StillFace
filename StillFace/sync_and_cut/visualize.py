import os
import argparse
from pathlib import Path
from typing import Dict, Optional
import subprocess
import cv2


DB_DIR = Path('data/ELTE-PPK_StillFace')


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of",
        "default=noprint_wrappers=1:nokey=1", str(video_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return 0.0


def stack_videos_vertical(video1: Path, video2: Path, output_video: Path, audio_source: int = 0) -> bool:
    """
    Stack two videos vertically using ffmpeg.
    
    Args:
        video1: Path to first video (top)
        video2: Path to second video (bottom)
        output_video: Path to output stacked video
        audio_source: Which video's audio to use (0 for video1, 1 for video2)
    
    Returns:
        True if successful, False otherwise
    """
    if not video1.exists() or not video2.exists():
        print(f"Warning: Cannot stack - missing video(s): {video1.name}, {video2.name}")
        return False
    
    if output_video.exists():
        print(f"Warning: Output video already exists: {output_video}")
        return True

    # Ensure output directory exists
    output_video.parent.mkdir(parents=True, exist_ok=True)
    
    # FFmpeg command to stack videos vertically with selected audio
    cmd = (
        f"ffmpeg -hide_banner -loglevel error -y "
        f"-i {video1} -i {video2} "
        f"-filter_complex '[0:v][1:v]vstack=inputs=2[v]' "
        f"-map '[v]' -map {audio_source}:a "
        f"{output_video}"
    )
    
    audio_from = "video2" if audio_source == 1 else "video1"
    print(f"Stacking {video1.name} + {video2.name} -> {output_video.name} (audio from {audio_from})")
    result = os.system(cmd)
    
    return result == 0


def stack_videos_2x2(
    video_paths: Dict[str, Optional[Path]],
    output_video: Path,
    audio_source: str = "baby"
) -> bool:
    """
    Stack videos in 2x2 grid: mother (top-left), window (top-right), baby (bottom-left), door (bottom-right).
    Missing videos are replaced with black frames.
    
    Args:
        video_paths: Dict with keys "mother", "window", "baby", "door" mapping to video paths (or None)
        output_video: Path to output stacked video
        audio_source: Which video's audio to use ("mother", "baby", "window", or "door")
    
    Returns:
        True if successful, False otherwise
    """
    # Ensure output directory exists
    output_video.parent.mkdir(parents=True, exist_ok=True)
    
    # Get reference video for dimensions and duration (prefer baby, then mother, then any available)
    ref_video = None
    for name in ["baby", "mother", "window", "door"]:
        if video_paths.get(name) and video_paths[name].exists():
            ref_video = video_paths[name]
            break
    
    if ref_video is None:
        print("Warning: No videos available for 2x2 stacking")
        return False
    
    # Build filter complex for 2x2 grid
    # Order: [0]=mother, [1]=window, [2]=baby, [3]=door
    inputs = []
    input_labels = []
    audio_map_idx = None
    video_exists = {}  # Track which videos actually exist
    
    for idx, name in enumerate(["mother", "window", "baby", "door"]):
        if video_paths.get(name) and video_paths[name].exists():
            inputs.append(f"-i {video_paths[name]}")
            input_labels.append(f"[{idx}:v]")
            video_exists[name] = idx
            if name == audio_source:
                audio_map_idx = idx
        else:
            # Create black video matching reference (no audio)
            inputs.append(f"-f lavfi -i color=black:size=1920x1080:rate=60:duration=300")
            input_labels.append(f"[{idx}:v]")
    
    # If audio source not found, use first available video with audio
    audio_priority = ["baby", "mother", "window", "door"]
    if audio_map_idx is None:
        for name in audio_priority:
            if name in video_exists:
                audio_map_idx = video_exists[name]
                audio_source = name
                break
    
    # Build filter: stack top row (mother, window), bottom row (baby, door), then stack vertically
    filter_complex = (
        f"{''.join(input_labels[:2])}hstack=inputs=2[top];"
        f"{''.join(input_labels[2:])}hstack=inputs=2[bottom];"
        f"[top][bottom]vstack=inputs=2[v]"
    )
    
    cmd = (
        f"ffmpeg -hide_banner -loglevel error -y "
        f"{' '.join(inputs)} "
        f"-filter_complex '{filter_complex}' "
        f"-map '[v]' "
    )
    
    # Only map audio if we have at least one real video with audio
    if audio_map_idx is not None:
        cmd += f"-map {audio_map_idx}:a? "
    
    cmd += f"-shortest {output_video}"
    
    if output_video.exists():
        print(f"Warning: Output video already exists: {output_video}")
        return True
    
    audio_info = f"audio from {audio_source}" if audio_map_idx is not None else "no audio"
    print(f"Creating 2x2 grid -> {output_video.name} ({audio_info})")
    result = os.system(cmd)
    
    return result == 0


def create_stacked_videos(
    cut_videos: Dict[str, Dict[str, Path]],
    output_dir: Path
):
    """
    Create stacked videos for each phase:
    1. mother-baby horizontal stack
    2. 2x2 grid: mother (top-left), window (top-right), baby (bottom-left), door (bottom-right)
    
    Args:
        cut_videos: Dict mapping video_name -> phase -> output_path
        output_dir: Directory to save stacked videos
    """
    phases = ["baseline", "play", "stillface", "reunion"]
    
    for phase in phases:
        # 1. Create mother-baby horizontal stack
        if ("mother" in cut_videos and phase in cut_videos["mother"] and
            "baby" in cut_videos and phase in cut_videos["baby"]):
            
            mother_path = cut_videos["mother"][phase]
            baby_path = cut_videos["baby"][phase]
            output_video = output_dir / f"mother-baby_{phase}.mp4"
            
            # Use baby's audio for mother-baby (mother on top, baby on bottom)
            stack_videos_vertical(mother_path, baby_path, output_video, audio_source=1)
        else:
            print(f"Skipping mother-baby for {phase} - video(s) not available")
        
        # 2. Create 2x2 grid
        video_paths = {}
        for name in ["mother", "window", "baby", "door"]:
            if name in cut_videos and phase in cut_videos[name]:
                video_paths[name] = cut_videos[name][phase]
            else:
                video_paths[name] = None
        
        output_video_2x2 = output_dir / f"session_{phase}.mp4"
        stack_videos_2x2(video_paths, output_video_2x2, audio_source="baby")


def find_cut_videos(db_dir: Path, session_id: str) -> Dict[str, Dict[str, Path]]:
    """
    Find cut videos for a specific session.
    
    Args:
        db_dir: Database directory
        session_id: Session ID
    
    Returns:
        Dict mapping video_name -> phase -> output_path
    """
    cut_dir = db_dir / "Sessions" / session_id / "processed"
    
    cut_videos = {}
    
    for video_name in ["mother", "baby", "window", "door"]:
        cut_videos[video_name] = {}
        
        for phase in ["baseline", "play", "stillface", "reunion"]:
            cut_video = cut_dir / f"{video_name}_{phase}.mp4"
            if cut_video.exists():
                cut_videos[video_name][phase] = cut_video
            else:
                print(f"Warning: Cut video not found: {cut_video}")
    
    return cut_videos


def generate_thumbnails(dir_base: Path, session_id: str = None):
    root_dir = dir_base / 'Sessions'
    session_ids = sorted([elem.name for elem in root_dir.glob('*')]) if session_id is None else [session_id]
    overview_dir = dir_base / 'Thumbnails'
    overview_dir.mkdir(parents=True, exist_ok=True)
    
    for session_id in session_ids:
        video_path = root_dir / session_id / 'visualize' / 'session_stillface.mp4'

        if video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f'Failed to open video for session {session_id}')
                continue
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
            ret, frame = cap.read()
            cap.release()

            if ret:
                save_path = root_dir / session_id / 'thumbnail.png'
                cv2.imwrite(str(save_path), frame)
                overview_path = overview_dir / f'{session_id}.png'
                cv2.imwrite(str(overview_path), frame)
                print(f'[INFO] Saved thumbnail for session {session_id}')
            else:
                print(f'[ERROR] Failed to read frame for session {session_id}')
        else:
            print(f'[ERROR] Video not found for session {session_id}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create stacked videos for each phase")
    parser.add_argument("--mode", type=str, default="thumbnail", help="Mode (stack, thumbnail)")
    parser.add_argument("--db_dir", type=Path, default=DB_DIR, help="Database directory")
    parser.add_argument("--session_id", type=str, default=None, help="Session ID")
    args = parser.parse_args()
    
    if args.mode == "stack":
        cut_videos = find_cut_videos(args.db_dir, args.session_id)
        create_stacked_videos(cut_videos, args.db_dir / "Sessions" / args.session_id / "visualize")
    elif args.mode == "thumbnail":
        generate_thumbnails(args.db_dir, args.session_id)