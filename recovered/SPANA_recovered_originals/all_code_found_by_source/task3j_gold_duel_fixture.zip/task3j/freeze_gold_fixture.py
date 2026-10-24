"""Task 3J: freeze GOLD_DUEL_01. Builds the guard-loadable BenchmarkFixture
(hashing/probing the extract via the only supported constructor), adds the
temporal fields, writes the traceability wrapper, runs both guards. No SPANA
tracking or event inference is run here."""
import json, hashlib
from dataclasses import replace, asdict
from event_engine_v1.benchmark_fixture import (BenchmarkFixture, validate_provenance,
    validate_temporal_coverage, evaluation_gate, sha256_file)
D = 'event_engine_v1/task3j/gold/'
EXTRACT = '/mnt/user-data/uploads/Downloads/_spana_task3j_work/gold/GOLD_DUEL_01_syo_mar_3036_3048.mp4'
label = json.load(open(D + 'GOLD_DUEL_HUMAN_LABEL.json'))
label_sha = sha256_file(D + 'GOLD_DUEL_HUMAN_LABEL.json')
SEED_FRAME = 180
SEED_BBOX = [635.3, 304.1, 757.8, 438.3]
fx = BenchmarkFixture.create(
    clip_id='gold_duel_01_syo_mar_3036_3048', video_path=EXTRACT,
    target_player_label='SOL #23 (royal blue), attacker',
    seed_frame=SEED_FRAME, seed_bbox=SEED_BBOX,
    expected_primary_event='DUEL', expected_outcome='WON',
    expected_event_window='7.30-9.67 s extract video time (source 3043.30-3045.67 s)',
    fixture_provenance_notes=('Task 3J gold duel. Extract = lossless (libx264 qp0, yuv420p) re-encode of source '
        'videoplayback.mp4 frames 91080-91439; all 360 frames verified bit-identical (framemd5) to the decoded source. '
        f'Human label file sha256 {label_sha}. Seed chosen by visual identification of #23, confirmed by raw YOLOv8x '
        'person detection (conf 0.939) at the seed frame.'))
fx = replace(fx, label_window_start_s=label['label_window_start_s'], label_window_end_s=label['label_window_end_s'],
             label_window_source='GOLD_DUEL_HUMAN_LABEL.json v1 (video time of the extract)',
             tracking_start_s=SEED_FRAME / 30.0, tracking_end_s=fx.duration_seconds,
             required_pre_event_context_s=0.5)
fx.save(D + 'GOLD_DUEL_01_BENCHMARK_FIXTURE.json')
loaded = BenchmarkFixture.load(D + 'GOLD_DUEL_01_BENCHMARK_FIXTURE.json')
prov = validate_provenance(loaded, EXTRACT); tc = validate_temporal_coverage(loaded); gate = evaluation_gate(loaded, EXTRACT)
wrapper = {
  'fixture_id': 'GOLD_DUEL_01', 'fixture_version': 1, 'clip_id': loaded.clip_id,
  'source_path': '/Users/nathangeorgaklis/Downloads/videoplayback.mp4',
  'source_sha256': '09aecc9b1ca2f64203864f25705618b03f35fa85cafd9610c51bdeecd585bfec',
  'source_registry_id': '6fdf7094bbc54b7da9148fd770165c53 (npl_nsw_syo_vs_mar)',
  'source_start_s': 3036.0, 'source_end_s': 3048.0, 'source_start_frame': 91080, 'source_end_frame_exclusive': 91440,
  'source_fps': 30.0, 'match_clock_at_extract_start': '27:16 (1st half), SOL 0-0 MAR',
  'extract_method': 'ffmpeg -ss 3036.0 -i videoplayback.mp4 -frames:v 360 -an -c:v libx264 -qp 0 -preset veryfast -pix_fmt yuv420p -color_range tv -r 30',
  'extract_frame_identity_check': 'framemd5 of all 360 extract frames == decoded source frames 91080-91439 (360/360)',
  'video_path': EXTRACT, 'video_path_mac': '/Users/nathangeorgaklis/Downloads/_spana_task3j_work/gold/GOLD_DUEL_01_syo_mar_3036_3048.mp4',
  'video_sha256': loaded.video_sha256, 'file_size': loaded.file_size_bytes, 'width': loaded.width, 'height': loaded.height,
  'fps': loaded.fps, 'frame_count': loaded.frame_count, 'duration': loaded.duration_seconds,
  'target_identity': label['target'], 'opponent_identity': label['opponent'],
  'target_team': 'SOL', 'target_shirt_number': '23',
  'seed_frame': SEED_FRAME, 'seed_time_s': SEED_FRAME / 30.0, 'seed_bbox': SEED_BBOX,
  'expected_primary_event': 'DUEL', 'expected_outcome': 'WON',
  'label_window_start_s': loaded.label_window_start_s, 'label_window_end_s': loaded.label_window_end_s,
  'required_pre_event_context_s': 0.5,
  'planned_tracking_start_s': loaded.tracking_start_s, 'planned_tracking_end_s': loaded.tracking_end_s,
  'tracking_note': 'Task 3K must re-run validate_temporal_coverage with the ACTUAL tracked range (tracking_range_from_mo) before scoring.',
  'human_description': label['human_description'], 'human_reviewer': label['human_reviewer'],
  'label_timestamp': label['human_label_created_at'], 'human_signoff': label['human_signoff'],
  'human_label_file': 'GOLD_DUEL_HUMAN_LABEL.json', 'human_label_sha256': label_sha,
  'guard_loadable_fixture_file': 'GOLD_DUEL_01_BENCHMARK_FIXTURE.json',
  'guard_loadable_fixture_sha256': sha256_file(D + 'GOLD_DUEL_01_BENCHMARK_FIXTURE.json'),
  'guards': {'provenance': json.loads(prov.to_json()), 'temporal': json.loads(tc.to_json()), 'evaluation_gate': json.loads(gate.to_json())},
  'frozen_before_inference': True, 'spana_inference_run': False,
}
json.dump(wrapper, open(D + 'GOLD_DUEL_FIXTURE.json', 'w'), indent=2)
print('provenance', prov.status, '| temporal', tc.status, tc.values, '| gate', json.loads(gate.to_json()))
