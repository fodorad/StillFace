import os
import shutil
import argparse
from pathlib import Path
from typing import Optional, Dict, Tuple
import pandas as pd
from avst.sync import sync_videos, sync_to_reference
from avst.io import get_video_fps


DB_DIR = Path('data/ELTE-PPK_StillFace')
CAMERA_NAMES = ['mother', 'baby', 'window', 'door']
PHASES = ['baseline', 'play', 'stillface', 'reunion']


def convert_to_60fps(input_path: Path, output_path: Path) -> Path:
    """Convert video to 60 FPS MP4 format."""
    if output_path.exists():
        return output_path
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = f"ffmpeg -hide_banner -loglevel error -y -i {input_path} -r 60 {output_path}"
    os.system(cmd)
    print(f"Converted to 60 FPS: {output_path}")
    return output_path


def get_available_cameras(session_dir: Path) -> Dict[str, Optional[Path]]:
    """
    Detect available camera videos in the session directory.
    
    Returns:
        Dict mapping camera name to video path (or None if missing)
    """
    original_dir = session_dir / 'original'
    cameras = {}
    
    # Check for each camera with expected file names
    camera_files = {
        'mother': original_dir / 'mother.mp4',
        'baby': original_dir / 'baby.mp4',
        'window': original_dir / 'window.MTS',
        'door': original_dir / 'door.MTS'
    }
    
    for cam_name, cam_path in camera_files.items():
        cameras[cam_name] = cam_path if cam_path.exists() else None
    
    return cameras


def prepare_original_videos(session_dir: Path, cameras: Dict[str, Optional[Path]]) -> Dict[str, Optional[Path]]:
    """
    Prepare original videos: convert MTS files to 60 FPS MP4.
    
    Returns:
        Updated dict with converted video paths
    """
    original_dir = session_dir / 'original'
    original_dir.mkdir(parents=True, exist_ok=True)
    
    prepared = cameras.copy()
    
    # Convert window.MTS to window.mp4 if needed
    if cameras['window'] is not None:
        window_mts = cameras['window']
        window_mp4 = original_dir / 'window.mp4'
        
        if abs(get_video_fps(window_mts) - 60) > 0.01:
            prepared['window'] = convert_to_60fps(window_mts, window_mp4)
        else:
            if not window_mp4.exists():
                shutil.copy(window_mts, window_mp4)
            prepared['window'] = window_mp4
    
    # Convert door.MTS to door.mp4 if needed
    if cameras['door'] is not None:
        door_mts = cameras['door']
        door_mp4 = original_dir / 'door.mp4'
        
        if abs(get_video_fps(door_mts) - 60) > 0.01:
            prepared['door'] = convert_to_60fps(door_mts, door_mp4)
        else:
            if not door_mp4.exists():
                shutil.copy(door_mts, door_mp4)
            prepared['door'] = door_mp4
    
    return prepared


def sync_mother_baby(
    session_dir: Path,
    mother_path: Optional[Path],
    baby_path: Optional[Path],
    visualize: bool = False
) -> Tuple[Dict[str, Optional[Path]], Optional[int]]:
    """
    Sync mother and baby videos.
    
    Returns:
        Tuple with dict with 'mother' and 'baby' keys mapping to synced video paths and the offset in ms
    """
    synced_dir = session_dir / 'synced'
    synced_dir.mkdir(parents=True, exist_ok=True)
    
    synced_mother = synced_dir / 'mother.mp4'
    synced_baby = synced_dir / 'baby.mp4'
    synced_session_mb = synced_dir / 'session_mb.mp4'
    
    result = {'mother': None, 'baby': None}
    ms_offset = None

    # Case 1: Both available - sync them
    if mother_path is not None and baby_path is not None:
        print("Syncing mother and baby videos...")
        ms_offset = sync_videos(
            video1_path=mother_path,
            video2_path=baby_path,
            output1_path=synced_mother,
            output2_path=synced_baby,
            synced_session_path=synced_session_mb if visualize else None,
        )
        result['mother'] = synced_mother
        result['baby'] = synced_baby

    # Case 2: Only baby available - copy it
    elif baby_path is not None:
        print("Copying baby video (no mother video available)...")
        shutil.copy(baby_path, synced_baby)
        result['baby'] = synced_baby
    
    # Case 3: Only mother available - copy it
    elif mother_path is not None:
        print("Copying mother video (no baby video available)...")
        shutil.copy(mother_path, synced_mother)
        result['mother'] = synced_mother
    
    return result, ms_offset


def sync_auxiliary_camera(
    session_dir: Path,
    cam_name: str,
    cam_path: Path,
    reference_path: Optional[Path],
    visualize: bool = False
) -> Tuple[Optional[Path], Optional[int]]:
    """
    Sync auxiliary camera (window or door) to reference video.
    
    Returns:
        Tuple with path to synced video or None and the offset in ms
    """
    synced_dir = session_dir / 'synced'
    synced_dir.mkdir(parents=True, exist_ok=True)
    
    synced_cam = synced_dir / f'{cam_name}.mp4'
    synced_session = synced_dir / f'session_{reference_path.stem}_{cam_name}.mp4'
    
    ms_offset = None

    if reference_path is None or cam_path == reference_path:
        # No reference available, just copy
        print(f"Copying {cam_name} video (no reference available)...")
        shutil.copy(cam_path, synced_cam)
        return synced_cam, ms_offset
    
    print(f"\nSyncing {cam_name} to reference video...")
    ms_offset = sync_to_reference(
        video1_path=reference_path,
        video2_path=cam_path,
        output2_path=synced_cam,
        synced_session_path=synced_session if visualize else None,
    )
    
    return synced_cam, ms_offset


def sync(
    db_dir: Path,
    session_id: str,
    visualize: bool = False
) -> Tuple[Dict[str, Optional[Path]], Optional[int]]:
    """
    Sync all available camera videos for a session.
    
    Returns:
        Tuple with dict with 'mother' and 'baby' keys mapping to synced video paths and the offset in ms
    
    Args:
        db_dir: Database directory (e.g., 'data/ELTE-PPK_StillFace')
        session_id: Session ID (e.g., '686527')
        visualize: Whether to create visualization videos
    """
    session_dir = db_dir / "Sessions" / session_id
    
    print(f"\n=== Syncing session: {session_id} ===")
    
    # Step 1: Detect available cameras
    cameras = get_available_cameras(session_dir)
    available = [name for name, path in cameras.items() if path is not None]
    print(f"Available cameras: {', '.join(available) if available else 'None'}")
    
    if not available:
        print("No cameras available. Exiting.")
        return
    
    # Step 2: Prepare original videos (convert MTS to MP4 at 60 FPS)
    cameras = prepare_original_videos(session_dir, cameras)
    
    # Step 3: Sync mother and baby
    synced, ms_offset_mb = sync_mother_baby(
        session_dir,
        cameras['mother'],
        cameras['baby'],
        visualize
    )
    
    # Step 4: Determine reference video for auxiliary cameras
    # Priority: baby > mother > window
    reference = None
    if synced['baby'] is not None:
        reference = synced['baby']
    elif synced['mother'] is not None:
        reference = synced['mother']
    elif cameras['window'] is not None:
        reference = cameras['window']
    
    # Step 5: Sync window camera
    if cameras['window'] is not None:
        synced['window'], _ = sync_auxiliary_camera(
            session_dir,
            'window',
            cameras['window'],
            reference,
            visualize
        )

        # Update reference if window is the first available
        if reference is None:
            reference = synced['window']
    
    # Step 6: Sync door camera
    if cameras['door'] is not None:
        synced['door'], _ = sync_auxiliary_camera(
            session_dir,
            'door',
            cameras['door'],
            reference,
            visualize
        )
    
    print(f"\n=== Sync complete for session: {session_id} ===")
    synced_available = [name for name, path in synced.items() if path is not None]
    print(f"Synced cameras: {', '.join(synced_available)}")

    return synced, ms_offset_mb


def is_synced(session_id: str) -> bool:
    synced_sessions_path = DB_DIR / "synced_sessions.txt"
    if not synced_sessions_path.exists():
        return False
    with open(synced_sessions_path, "r") as f:
        for line in f:
            if line.startswith(str(session_id)):
                return True
    return False


def sync_all(
    db_dir: Path,
    metadata_path: Path,
    visualize: bool = False
):
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    metadata = pd.read_excel(metadata_path)
    
    for index, row in metadata.iterrows():

        if is_synced(str(row['ID'])): continue
        if row['Auto'] == 'n': continue
        if not pd.isna(row['offset_mother-baby_(ms)']): continue

        try:
            _, ms_offset_mb = sync(
                db_dir=db_dir,
                session_id=str(row['ID']),
                visualize=visualize
            )
        except Exception as e:
            with open(DB_DIR / "failed_sessions.txt", "a") as f:
                f.write(str(row['ID']) + "\n")
            continue

        with open(DB_DIR / "synced_sessions.txt", "a") as f:
            f.write(str(row['ID']) + "," + str(ms_offset_mb) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync video files for StillFace sessions")
    parser.add_argument("--db_dir", type=Path, default=DB_DIR, help="Database directory")
    parser.add_argument("--metadata_path", type=Path, default=DB_DIR / "metadata_database.xlsx", help="Metadata file path")
    parser.add_argument("--session_id", type=str, default=None, help="Session ID")
    parser.add_argument("--visualize", action="store_true", help="Create visualization videos")
    args = parser.parse_args()
    
    if args.session_id is not None:
        sync(
            db_dir=args.db_dir,
            session_id=args.session_id,
            visualize=args.visualize
        )
    elif Path(args.metadata_path).exists():
        sync_all(
            db_dir=args.db_dir,
            metadata_path=args.metadata_path,
            visualize=args.visualize
        )
