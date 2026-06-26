import json
import sqlite3
from pathlib import Path
import ffmpeg


BATCH_SIZE = 1000

def init_database(db_path):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS session (
            session_id TEXT PRIMARY KEY,
            start_time BIGINT,
            frame_count INTEGER,
            fps REAL,
            other_metadata JSON,
            video_usable BOOLEAN,
            audio_in_usable BOOLEAN,
            audio_out_usable BOOLEAN,
            mouse_usable BOOLEAN,
            keyboard_usable BOOLEAN
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS frame (
            frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT REFERENCES session(session_id),
            start_timestamp BIGINT
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS mouse_click (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT REFERENCES session(session_id),
            mouse_key_type TEXT,
            start_timestamp BIGINT,
            end_timestamp BIGINT
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS mouse_movement (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT REFERENCES session(session_id),
            timestamp BIGINT,
            mouseX REAL,
            mouseY REAL,
            mouseDX REAL,
            mouseDY REAL
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS keyboard (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT REFERENCES session(session_id),
            key_id TEXT,
            start_timestamp BIGINT,
            end_timestamp BIGINT
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transcript_in (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT REFERENCES session(session_id),
            word TEXT,
            start_time BIGINT,
            end_time BIGINT
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transcript_out (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT REFERENCES session(session_id),
            word TEXT,
            start_time BIGINT,
            end_time BIGINT
        )""")
    # Removed the single speaker_diarization table
    # Added two separate tables for speaker diarization
    cur.execute("""
        CREATE TABLE IF NOT EXISTS speaker_diarization_in (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT REFERENCES session(session_id),
            speaker_label TEXT,
            start_time BIGINT,
            end_time BIGINT
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS speaker_diarization_out (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT REFERENCES session(session_id),
            speaker_label TEXT,
            start_time BIGINT,
            end_time BIGINT
        )""")
    con.commit()
    return con, cur



def create_session_folders(session_id, output_path):
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    paths = {
        "db": output_path / f"{session_id}.db",
        "video": output_path / f"{session_id}_video.mp4",
        "audio_in": output_path / f"{session_id}_audio_in.wav",
        "audio_out": output_path / f"{session_id}_audio_out.wav",
        "position_data": output_path / f"{session_id}_position_data.json",
        "eventlog": output_path / f"{session_id}_eventlog.json"
    }
    
    return paths

def get_trim_start(keyboard_data):
    first_event_timestamp = min(event['timestamp'] for event in keyboard_data)
    return first_event_timestamp + 5000

def get_trim_end(keyboard_data, frame_start_timestamp, video_duration, fallback_trim=15000):
    first_of_the_last_esc_presses = None

    for i in range(len(keyboard_data) - 1, -1, -1):
        event = keyboard_data[i]
        
        if event['key'] != 256 and first_of_the_last_esc_presses is None:
            return video_duration - fallback_trim
        
        if event['key'] != 256 and first_of_the_last_esc_presses is not None:
            return first_of_the_last_esc_presses['timestamp'] - frame_start_timestamp - 1000

        if event['key'] == 256 and event['action'] == 'PRESS':
            first_of_the_last_esc_presses = event

    return video_duration - fallback_trim

def set_modality_flags_on_version(version):
    video_usable = True
    audio_in_usable = True
    audio_out_usable = True
    mouse_usable = True
    keyboard_usable = True
    
    if version < 8:
        print("[VERSION] < 8 audio is not usable")
        audio_in_usable = False
        audio_out_usable = False
    
    if version < 5:
        print("[VERSION] < 5 mouse and keyboard are not usable")
        mouse_usable = False
        keyboard_usable = False
    return video_usable, audio_in_usable, audio_out_usable, mouse_usable, keyboard_usable



def process_mouse_click_data(mouse_clicks, con, cur, session_id, trim_start, end_time_unix):
    mouse_data = json.loads(mouse_clicks)
    processed_events = []
    button_press_times = {'LEFT': None, 'RIGHT': None, 'MIDDLE': None}

    print(f"[DEBUG] Session {session_id}: Processing mouse click data with trim_start={trim_start}, end_time_unix={end_time_unix}")

    for event in mouse_data:
        action = event['action']
        timestamp = event['timestamp']
        
        if timestamp < trim_start or timestamp > end_time_unix:
            continue

        # adjusted_timestamp = timestamp - trim_start
        adjusted_timestamp = timestamp

        if 'PRESS' in action:
            button = action.split('_')[0]
            button_press_times[button] = adjusted_timestamp
        elif 'RELEASE' in action:
            button = action.split('_')[0]
            if button_press_times[button] is not None:
                processed_events.append({
                    'key': button.lower(),
                    'start_time': button_press_times[button],
                    'end_time': adjusted_timestamp
                })
                button_press_times[button] = None
        elif 'SCROLL' in action:
            scroll_type = 'scroll_up' if 'UP' in action else 'scroll_down'
            processed_events.append({
                'key': scroll_type,
                'start_time': adjusted_timestamp,
                'end_time': adjusted_timestamp
            })

    print(f"[DEBUG] Session {session_id}: processed_events count={len(processed_events)}")

    for j, event in enumerate(processed_events):
        cur.execute("""
            INSERT INTO mouse_click (session_id, mouse_key_type, start_timestamp, end_timestamp)
            VALUES (?, ?, ?, ?)
        """, (session_id, event['key'], int(event['start_time']), int(event['end_time'])))

        if (j + 1) % BATCH_SIZE == 0:
            con.commit()

    con.commit()

def process_mouse_movement_and_frame(con, cur, session_id, frame_count, frame_duration, adjusted_mouse_data, trim_start):
    for event in adjusted_mouse_data:
        ###event['timestamp'] -= trim_start
        cur.execute("""
            INSERT INTO mouse_movement (session_id, timestamp, mouseX, mouseY, mouseDX, mouseDY)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, event['timestamp'], event['mouseX'], event['mouseY'], event['mouseDX'], event['mouseDY']))

    for i in range(frame_count):
        frame_start_time = int(i * frame_duration)
        cur.execute("""
            INSERT INTO frame (session_id, start_timestamp)
            VALUES (?, ?)
        """, (session_id, frame_start_time))
        
        if (i + 1) % BATCH_SIZE == 0:
            con.commit()

    con.commit()

def process_keyboard_data(keyboard_data, con, cur, session_id, trim_start, end_time_unix):
    processed_events = []
    key_press_times = {}

    print(f"[DEBUG] Session {session_id}: Processing keyboard data with trim_start={trim_start}, end_time_unix={end_time_unix}")

    for event in keyboard_data:
        key_id = event['key']
        timestamp = event['timestamp']

        if timestamp < trim_start or timestamp > end_time_unix:
            continue

        ###adjusted_timestamp = timestamp - trim_start
        adjusted_timestamp = timestamp
        action = event['action']

        if action == 'PRESS':
            if key_id not in key_press_times:
                key_press_times[key_id] = adjusted_timestamp
        elif action == 'RELEASE':
            if key_id in key_press_times:
                processed_events.append({
                    'key': key_id,
                    'start_time': key_press_times[key_id],
                    'end_time': adjusted_timestamp
                })
                del key_press_times[key_id]

    print(f"[DEBUG] Session {session_id}: processed_events count={len(processed_events)}")

    for j, event in enumerate(processed_events):
        cur.execute("""
            INSERT INTO keyboard (session_id, key_id, start_timestamp, end_timestamp)
            VALUES (?, ?, ?, ?)
        """, (session_id, event['key'], int(event['start_time']), int(event['end_time'])))

        if (j + 1) % BATCH_SIZE == 0:
            con.commit()

    con.commit()

def save_session_data(cur, con, session_id, start_time, frame_count, fps, other_metadata, video_usable, audio_in_usable, audio_out_usable, mouse_usable, keyboard_usable):
    cur.execute("""
        INSERT OR REPLACE INTO session (session_id, start_time, frame_count, fps, other_metadata, video_usable, audio_in_usable, audio_out_usable, mouse_usable, keyboard_usable)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (session_id, start_time, frame_count, fps, json.dumps(other_metadata), video_usable, audio_in_usable, audio_out_usable, mouse_usable, keyboard_usable))
    con.commit()

def next_numbered_path(base_path):
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    while True:
        candidate = base.parent / f"{base.stem}_{n}{base.suffix}"
        if not candidate.exists():
            return candidate
        n += 1

def process_session_data(mousedata: str, clickdata: str, keydata: str, video_path: Path, session_id: str, output_path: Path):
    ##video_path = Path(data_paths['video'])
    ##audio_path = Path(data_paths['audio'])
    
    session_paths = create_session_folders(session_id, output_path)
    '''
    with open(data_paths['info'], 'r') as file:
        info_data = json.load(file)
        version = int(info_data.get("version", 0))
        if version == 5:
            keyboard_data_reconstruct(Path(data_paths['info']).parent, GLFW_MAPPINGS_PATH)
        if 5 <= version:
            mouse_data_reconstruct(Path(data_paths['info']).parent)

         '''   
    keyboard_data = json.loads(keydata)
    mouse_data = json.loads(mousedata)


    '''
    trim_start = get_trim_start(keydata)
    frame_start_timestamp = int(video_path.stem)

    if trim_start > frame_start_timestamp:
        trim_duration = trim_start - frame_start_timestamp
    else:
        trim_duration = 0
    '''

    version = 5
    con, cur = init_database(session_paths["db"])
    trim_start= min(event['timestamp'] for event in mouse_data)
    end_time_unix = max(event['timestamp'] for event in mouse_data)
    
    video_usable, audio_in_usable, audio_out_usable, mouse_usable, keyboard_usable = set_modality_flags_on_version(version)


    
    try:
        start_time = trim_start
        end_time_unix = end_time_unix

        if end_time_unix <= start_time:
            print(f"Video duration is invalid after trimming. Skipping this session.")
            video_usable = False

        if video_usable:
            out_path = next_numbered_path(session_paths["video"])
            try:
                #TODO: CHANGE THIS TO NVIDIA ENCODER FOR THE REAL RUN
                #output(str(session_paths["video"]), vcodec='h264_nvenc')\
                ffmpeg.input(str(video_path))\
                  .output(str(out_path), vcodec='libx264')\
                  .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            except ffmpeg.Error as e:
                print(f"ffmpeg error while trimming video file {video_path}:")
                print(f"stdout: {e.stdout.decode('utf-8')}")
                print(f"stderr: {e.stderr.decode('utf-8')}")
                video_usable = False
                raise RuntimeError(f"Failed to process video for session {session_id}") from e

    except ffmpeg.Error as e:
        print(f"ffmpeg probe error for video file {video_path}: {e.stderr.decode('utf-8')}")
        video_usable = False
        raise RuntimeError(f"Failed to probe video for session {session_id}") from e
    except Exception as e:
        print(f"Error extracting frame count and FPS from video file {video_path}: {e}")
        video_usable = False
        raise RuntimeError(f"Failed to extract frame count and FPS for session {session_id}") from e
    

    if not video_usable:
        con.close()
        raise RuntimeError(f"Video processing failed for session {session_id}")
    
    '''
    process_audio(audio_path, start_time, end_time, session_paths, session_id, con, cur, audio_in_usable, audio_out_usable)

    other_metadata = {}
    with open(data_paths['info'], 'r') as file:
        other_metadata = json.load(file)
    '''

    frame_count, fps = 300, 30
    frame_duration = 1000 / fps if fps > 0 else 0

    #adjusted_mouse_data = [event for event in mouse_data if trim_start <= event['timestamp'] <= end_time_unix]
    adjusted_mouse_data = mouse_data.copy()


    process_mouse_click_data(clickdata, con, cur, session_id, trim_start, end_time_unix)
    process_mouse_movement_and_frame(con, cur, session_id, frame_count, frame_duration, adjusted_mouse_data, trim_start)

    if keyboard_usable:
        process_keyboard_data(keyboard_data, con, cur, session_id, trim_start, end_time_unix)
    
    other_metadata = {}
    save_session_data(cur, con, session_id, trim_start, frame_count, fps, other_metadata, video_usable, audio_in_usable, audio_out_usable, mouse_usable, keyboard_usable)

    con.commit()
    ###shutil.copyfile(data_paths['position'], session_paths["position_data"])
    ###shutil.copyfile(data_paths['eventlog'], session_paths["eventlog"])

    return con, cur, video_path, trim_start, end_time_unix





def preprocess_data(mousedata: str, clickdata: str, keydata: str, videopath: Path, outputpath: str):             
    video_path = videopath
    output_path = Path(outputpath)
    session_id = video_path.parent.name
    output_path = output_path / session_id
    print("asdf")

    con, cur, video_path, trim_start, end_time_unix = process_session_data(mousedata, clickdata, keydata, video_path, session_id, output_path)

    if trim_start is None or end_time_unix is None:
        print(f"Failed to process session: {session_id}")
        raise RuntimeError(f"Failed to process session: {session_id}")
    else:
        print(f"Successfully processed session: {session_id} and written to {output_path}")
        print(f"Trim start: {trim_start}")
        print(f"End time (Unix): {end_time_unix}")

    return con, cur, video_path