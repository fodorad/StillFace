import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from StillFace.sync_and_cut.visualize import create_stacked_videos


DB_DIR = Path('data/ELTE-PPK_StillFace')


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
    
    if output_video.exists():
        print(f"Warning: Output video already exists: {output_video}")
        return True
    
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

    # check input video mp4 length in seconds
    input_video_length = os.popen(f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {input_video}").read()
    input_video_length = float(input_video_length)

    # check that start_seconds are within input video length
    if start_seconds > input_video_length:
        print(f"Warning: Start time {start_seconds} is outside of input video length {input_video_length}")
        return False
    
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


def cut(
    db_dir: Path,
    session_id: str,
    baseline_timestamps: str,
    play_timestamps: str,
    stillface_timestamps: str,
    reunion_timestamps: str,
    visualize: bool = False
):
    """
    Main function to cut videos by phase and create stacked versions.
    
    Args:
        db_dir: Directory containing synced videos
        timestamp_file: File containing phase timestamps
        output_dir: Directory to save cut and stacked videos
    """
    # Create output directory
    output_dir = db_dir / "Sessions" / session_id / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    visualize_dir = db_dir / "Sessions" / session_id / "visualize"
    visualize_dir.mkdir(parents=True, exist_ok=True)
    synced_dir = db_dir / "Sessions" / session_id / "synced"
    
    # Cut all videos into phases
    print("\n=== Cutting videos by phase ===")
    cut_videos = cut_all_phases(synced_dir, baseline_timestamps, play_timestamps, stillface_timestamps, reunion_timestamps, output_dir)
    
    if visualize:
        # Create stacked videos
        print("\n=== Creating stacked videos ===")
        create_stacked_videos(cut_videos, visualize_dir)
    
    print(f"\n=== Done! Output saved to: {output_dir} ===")


def cut_all(
    db_dir: Path,
    metadata_path: Path,
    visualize: bool = False
):
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    if metadata_path.suffix == ".xlsx":
        metadata = pd.read_excel(metadata_path)
    elif metadata_path.suffix == ".csv":
        metadata = pd.read_csv(metadata_path)
    else:
        raise ValueError(f"Unsupported file format: {metadata_path.suffix}")
    
    for index, row in metadata.iterrows():

        if row['Auto'] == 'n': continue
        
        baseline_timestamps = f'{row["baseline_start_(MM:SS)"]}-{row["baseline_end_(MM:SS)"]}'
        play_timestamps = f'{row["play_start_(MM:SS)"]}-{row["play_end_(MM:SS)"]}'
        stillface_timestamps = f'{row["stillface_start_(MM:SS)"]}-{row["stillface_end_(MM:SS)"]}'
        reunion_timestamps = f'{row["reunion_start_(MM:SS)"]}-{row["reunion_end_(MM:SS)"]}'

        try:
            cut(
                db_dir=db_dir,
                session_id=str(row['ID']),
                baseline_timestamps=baseline_timestamps,
                play_timestamps=play_timestamps,
                stillface_timestamps=stillface_timestamps,
                reunion_timestamps=reunion_timestamps,
                visualize=visualize
            )
        except Exception as e:
            with open(DB_DIR / "failed_cut_sessions.txt", "a") as f:
                f.write(str(row['ID']) + "\n")
            continue

        with open(DB_DIR / "cut_sessions.txt", "a") as f:
            f.write(str(row['ID']) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut synced videos into phases and create stacked versions")
    parser.add_argument("--db_dir", type=Path, default=DB_DIR, help="Database directory")
    parser.add_argument("--metadata_path", type=Path, default=DB_DIR / "metadata.csv", help="Metadata file path")
    parser.add_argument("--session_id", type=str, default=None, help="Session ID")
    parser.add_argument("--visualize", action="store_true", help="Create visualization videos")
    args = parser.parse_args()
    
    if args.session_id is not None:
        metadata = pd.read_csv(args.metadata_path)
        df = metadata[metadata['ID'] == int(args.session_id)]
        baseline_timestamps = f'{df["baseline_start_(MM:SS)"].iloc[0]}-{df["baseline_end_(MM:SS)"].iloc[0]}'
        play_timestamps = f'{df["play_start_(MM:SS)"].iloc[0]}-{df["play_end_(MM:SS)"].iloc[0]}'
        stillface_timestamps = f'{df["stillface_start_(MM:SS)"].iloc[0]}-{df["stillface_end_(MM:SS)"].iloc[0]}'
        reunion_timestamps = f'{df["reunion_start_(MM:SS)"].iloc[0]}-{df["reunion_end_(MM:SS)"].iloc[0]}'
        cut(
            db_dir=args.db_dir,
            session_id=args.session_id,
            baseline_timestamps=baseline_timestamps,
            play_timestamps=play_timestamps,
            stillface_timestamps=stillface_timestamps,
            reunion_timestamps=reunion_timestamps,
            visualize=True
        )
    elif Path(args.metadata_path).exists():
        cut_all(
            db_dir=args.db_dir,
            metadata_path=args.metadata_path,
            visualize=True
        )
