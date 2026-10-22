import json
import subprocess
import time
import pytest
from types import SimpleNamespace

from fastapi.testclient import TestClient

import app


def test_highlights_are_nonoverlapping():
    segments = [dict(start=i, end=i + 5, text='중요한 핵심 방법입니다') for i in range(0, 120, 5)]
    clips = app.highlights(segments, 120, 30, 3)
    assert len(clips) == 3
    for a in clips:
        assert 0 <= a['start'] < a['end'] <= 120
        assert a['end'] - a['start'] == 30
        for b in clips:
            if a is not b:
                assert a['end'] <= b['start'] or a['start'] >= b['end']


def test_subtitles_are_clipped_and_sanitized():
    text = app.subtitles([dict(start=0, end=10, text='<b>한국어</b> {위치}자막')], 3, 6)
    assert '00:00:00,000 --> 00:00:03,000' in text
    assert '<b>' not in text and '{위치}' not in text
    assert '한국어' in text


@pytest.mark.parametrize('has_speech', [True, False])
def test_upload_render_and_download(tmp_path, monkeypatch, has_speech):
    """Use deterministic ASR output; real FFmpeg, multipart upload and file downloads."""
    monkeypatch.setattr(app, 'DATA', tmp_path)
    class FakeASR:
        def transcribe(self, *args, **kwargs):
            return iter([SimpleNamespace(start=0, end=3, text='중요한 핵심 장면입니다', words=[])] if has_speech else []), None
    monkeypatch.setattr(app, 'model', FakeASR())
    source = tmp_path / 'sample.mp4'
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=navy:s=320x240:d=4', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=4', '-c:v', 'libx264', '-c:a', 'aac', '-shortest', str(source)], check=True)
    with TestClient(app.app) as client:
        assert client.get('/').status_code == 200
        assert client.post('/api/jobs', files={'file': ('bad.mp4', b'not a video')}).status_code == 400
        assert client.post('/api/jobs', files={'file': ('x', b'x')}, data={'length': 9}).status_code == 400
        with source.open('rb') as f:
            response = client.post('/api/jobs', files={'file': ('sample.mp4', f, 'video/mp4')}, data={'length': 15, 'count': 1})
        assert response.status_code == 202
        job_id = response.json()['id']
        for _ in range(200):
            job = client.get('/api/jobs/' + job_id).json()
            if job['status'] != 'processing':
                break
            time.sleep(0.1)
        assert job['status'] == 'done', job
        clip = job['clips'][0]
        assert client.get(clip['video']).headers['content-type'] == 'video/mp4'
        if has_speech:
            assert '중요한' in client.get(clip['subtitle']).text
        else:
            assert clip['subtitle'] is None
            assert '말소리' in job['warning']
        info = json.loads(app.command(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(tmp_path / job_id / 'clip-0.mp4')]))
        video = next(s for s in info['streams'] if s['codec_type'] == 'video')
        assert (video['width'], video['height']) == (720, 1280)
        assert client.get(f'/api/jobs/{job_id}/files/source').status_code == 404


def test_recreation_rights_review_persistence_and_bundle(tmp_path, monkeypatch):
    import zipfile
    import io
    monkeypatch.setattr(app, 'DATA', tmp_path)
    class NarrationASR:
        def transcribe(self, *args, **kwargs):
            return iter([SimpleNamespace(start=0.5, end=7.5, text='직접 작성한 해설입니다. 새로운 관점으로 장면을 설명합니다.', words=[])]), None
    monkeypatch.setattr(app, 'model', NarrationASR())
    source = tmp_path / 'sample.mp4'
    voice = tmp_path / 'voice.wav'
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=blue:s=240x320:d=4', '-c:v', 'libx264', str(source)], check=True)
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', 'sine=frequency=400:duration=8', str(voice)], check=True)
    brief = dict(topic='화면 구성의 새로운 관점', viewpoint='화면 안의 색상 배치가 전달하는 효과를 분석합니다.', facts='테스트 영상은 직접 생성한 파란색 화면입니다.', conclusion='관찰 가능한 요소를 근거로 설명합니다.', credit='직접 생성한 테스트 영상', rights_basis='테스트용 직접 생성한 영상과 음원', script='직접 작성한 해설입니다. 새로운 관점으로 장면을 설명합니다.', rights_confirmed=True, script_confirmed=True, scenes=[dict(start=0, end=2, reason='화면의 색상 배치를 설명하기 위한 사례')])
    with TestClient(app.app) as client:
        draft = client.post('/api/recreation/draft', json={k: brief[k] for k in ['topic', 'viewpoint', 'facts', 'conclusion']})
        assert draft.status_code == 200
        assert brief['facts'] in draft.json()['script']
        assert '생성형 AI' in draft.json()['notice']
        def upload(payload):
            return client.post('/api/recreation', files={'file': ('sample.mp4', source.read_bytes(), 'video/mp4'), 'narration': ('voice.wav', voice.read_bytes(), 'audio/wav')}, data={'brief_json': json.dumps(payload)})
        assert upload({**brief, 'rights_confirmed': False}).status_code == 400
        assert upload({**brief, 'scenes': [dict(start=0, end=9, reason='원본 범위를 넘는 장면')]}).status_code == 400
        assert upload({**brief, 'scenes': [dict(start=0, end=4, reason='첫 장면을 설명합니다'), dict(start=0, end=2, reason='두 번째 장면 설명')]}).status_code == 400
        response = upload(brief)
        assert response.status_code == 202, response.text
        job_id = response.json()['id']
        for _ in range(600):
            job = client.get('/api/jobs/' + job_id).json()
            if job['status'] != 'processing':
                break
            time.sleep(0.1)
        assert job['status'] == 'done', job
        assert client.get(job['preview']).status_code == 200
        download = '/api/jobs/' + job_id + '/files/'
        assert client.get(download + 'recreation.mp4').status_code == 403
        checks = dict(watched=True, rights=True, facts=True, music=True, privacy=True, original_value=True)
        assert client.post(f'/api/recreation/{job_id}/review', json={**checks, 'facts': False}).status_code == 400
        result = client.post(f'/api/recreation/{job_id}/review', json=checks)
        assert result.status_code == 200
        assert client.get(result.json()['downloads']['video']).headers['content-type'] == 'video/mp4'
        assert '제작자 해설' in client.get(result.json()['downloads']['subtitle']).text
        archive = zipfile.ZipFile(io.BytesIO(client.get(result.json()['downloads']['bundle']).content))
        assert {'recreation.mp4', 'recreation.srt', 'script.txt', 'brief.json', 'review.json', '재창작_지침.md'} == set(archive.namelist())
        info = json.loads(app.command(['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(tmp_path / job_id / 'recreation.mp4')]))
        v = next(s for s in info['streams'] if s['codec_type'] == 'video')
        assert (v['width'], v['height']) == (720, 1280)
        assert abs(float(info['format']['duration']) - 8) < 0.1
        app.jobs.pop(job_id)
        assert client.get('/api/jobs/' + job_id).json()['reviewed'] is True
        assert client.get('/api/downloads/guide').status_code == 200
        program = zipfile.ZipFile(io.BytesIO(client.get('/api/downloads/program').content))
        assert 'moment-recreation/app.py' in program.namelist()
        assert not any('/data/' in n or '/.venv/' in n for n in program.namelist())


def test_recreation_rejects_narration_without_speech(tmp_path, monkeypatch):
    monkeypatch.setattr(app, 'DATA', tmp_path)
    class NoSpeech:
        def transcribe(self, *args, **kwargs):
            return iter([]), None
    monkeypatch.setattr(app, 'model', NoSpeech())
    job_id = 'f' * 32
    folder = tmp_path / job_id
    folder.mkdir()
    app.jobs[job_id] = dict(id=job_id, kind='recreation', status='processing', brief={}, duration=10, reviewed=False)
    app.render_recreation(job_id)
    assert app.jobs[job_id]['status'] == 'error'
    assert '말소리' in app.jobs[job_id]['stage']
    assert not (folder / 'recreation.mp4').exists()
