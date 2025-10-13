import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def cut_video(input_video: Path, output_video: Path, start_time: str, end_time: str) -> bool:
    """
    Cut video from start_time to end_time using ffmpeg.
    
    Args:
        input_video: Path to input video
        output_video: Path to output video
        start_time: Start time in MM:SS format
        end_time: End time in MM:SS format
    
    Returns:
        True if successful, False otherwise
    """
    if not input_video.exists():
        print(f"Warning: Input video not found: {input_video}")
        return False
    
    # Convert MM:SS to seconds for duration calculation
    def mmss_to_seconds(mmss: str) -> float:
        parts = mmss.split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return 0
    
    start_seconds = mmss_to_seconds(start_time)
    end_seconds = mmss_to_seconds(end_time)
    duration = end_seconds - start_seconds
    
    # Ensure output directory exists
    output_video.parent.mkdir(parents=True, exist_ok=True)
    
    # FFmpeg command to cut video
    cmd = (
        f"ffmpeg -hide_banner -loglevel error -y "
        f"-ss {start_time} -i {input_video} "
        f"-t {duration} "
        f"-c copy "
        f"{output_video}"
    )
    
    print(f"Cutting {input_video.name} -> {output_video.name} ({start_time} to {end_time})")
    result = os.system(cmd)
    
    return result == 0


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


def cut_all_phases(
    synced_dir: Path,
    baseline_timestamps: str,
    play_timestamps: str,
    stillface_timestamps: str,
    reunion_timestamps: str,
    output_dir: Path,
    video_names: List[str] = ["mother", "baby", "window", "door"],
    phases: List[str] = ["baseline", "play", "stillface", "reunion"]
) -> Dict[str, Dict[str, Path]]:
    """
    Cut all synced videos into phases.
    
    Args:
        synced_dir: Directory containing synced videos (mother.mp4, baby.mp4, etc.)
        timestamp_file: File containing phase timestamps
        output_dir: Directory to save cut videos
        video_names: List of video names to process
    
    Returns:
        Dict mapping video_name -> phase -> output_path
    """
    convert = lambda x: x.split("-")
    timestamps = {
        phase: convert(timestamps) 
        for phase, timestamps 
        in zip(phases, [baseline_timestamps, play_timestamps, stillface_timestamps, reunion_timestamps])
    }

    # Cut each video for each phase
    cut_videos = {}
    
    for video_name in video_names:
        input_video = synced_dir / f"{video_name}.mp4"
        
        if not input_video.exists():
            print(f"Skipping {video_name} - video not found: {input_video}")
            continue
        
        cut_videos[video_name] = {}
        
        for phase in phases:
            start_time, end_time = timestamps[phase]
            output_video = output_dir / f"{video_name}_{phase}.mp4"
            
            success = cut_video(input_video, output_video, start_time, end_time)
            
            if success:
                cut_videos[video_name][phase] = output_video
    
    return cut_videos


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
    
    for idx, name in enumerate(["mother", "window", "baby", "door"]):
        if video_paths.get(name) and video_paths[name].exists():
            inputs.append(f"-i {video_paths[name]}")
            input_labels.append(f"[{idx}:v]")
            if name == audio_source:
                audio_map_idx = idx
        else:
            # Create black video matching reference
            inputs.append(f"-f lavfi -i color=black:size=1920x1080:rate=60:duration=300")
            input_labels.append(f"[{idx}:v]")
    
    # If audio source not found, use first available video with audio
    if audio_map_idx is None:
        for idx, name in enumerate(["baby", "mother", "window", "door"]):
            if video_paths.get(name) and video_paths[name].exists():
                audio_map_idx = idx
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
    
    if audio_map_idx is not None:
        cmd += f"-map {audio_map_idx}:a "
    
    cmd += f"-shortest {output_video}"
    
    print(f"Creating 2x2 grid -> {output_video.name} (audio from {audio_source})")
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
            output_video = output_dir / 'visualize' / f"mother-baby_{phase}.mp4"
            
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
        
        output_video_2x2 = output_dir / 'visualize' / f"session_{phase}.mp4"
        stack_videos_2x2(video_paths, output_video_2x2, audio_source="baby")


def cut(
    synced_dir: Path,
    baseline_timestamps: str,
    play_timestamps: str,
    stillface_timestamps: str,
    reunion_timestamps: str,
    output_dir: Path
):
    """
    Main function to cut videos by phase and create stacked versions.
    
    Args:
        synced_dir: Directory containing synced videos
        timestamp_file: File containing phase timestamps
        output_dir: Directory to save cut and stacked videos
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cut all videos into phases
    print("\n=== Cutting videos by phase ===")
    cut_videos = cut_all_phases(synced_dir, baseline_timestamps, play_timestamps, stillface_timestamps, reunion_timestamps, output_dir)
    
    # Create stacked videos
    print("\n=== Creating stacked videos ===")
    create_stacked_videos(cut_videos, output_dir)
    
    print(f"\n=== Done! Output saved to: {output_dir} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut synced videos into phases and create stacked versions")
    parser.add_argument("--synced_dir", type=Path, required=True, help="Directory containing synced videos")
    parser.add_argument("--baseline_timestamps", type=str, required=True, help="Baseline timestamps")
    parser.add_argument("--play_timestamps", type=str, required=True, help="Play timestamps")
    parser.add_argument("--stillface_timestamps", type=str, required=True, help="Stillface timestamps")
    parser.add_argument("--reunion_timestamps", type=str, required=True, help="Reunion timestamps")
    parser.add_argument("--out_dir", type=Path, default='.', help="Output directory")
    args = parser.parse_args()
    
    cut(
        synced_dir=args.synced_dir,
        baseline_timestamps=args.baseline_timestamps,
        play_timestamps=args.play_timestamps,
        stillface_timestamps=args.stillface_timestamps,
        reunion_timestamps=args.reunion_timestamps,
        output_dir=args.out_dir
    )
