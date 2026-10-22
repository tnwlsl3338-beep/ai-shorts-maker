import json
import logging
import os
import re
import shutil
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)
os.environ.setdefault('HF_HOME', str(DATA / 'models'))
app = FastAPI(title='모먼트 · 한국어 AI 쇼츠')
app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')
pool = ThreadPoolExecutor(max_workers=1)
jobs = {}
model = None
model_lock = threading.Lock()


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=3600)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    return result.stdout


def probe(path):
    info = json.loads(command(['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(path)]))
    if not any(s['codec_type'] == 'video' for s in info['streams']):
        raise ValueError('영상 트랙이 있는 파일을 업로드해 주세요.')
    if not any(s['codec_type'] == 'audio' for s in info['streams']):
        raise ValueError('자동 자막 생성을 위해 음성이 포함된 영상이 필요합니다.')
    duration = float(info['format']['duration'])
    if not 3 <= duration <= 7200:
        raise ValueError('3초 이상 2시간 이하 영상을 사용해 주세요.')
    return duration


def highlights(segments, duration, length, count):
    """Rank speech windows by Korean hook phrases and speaking density."""
    cues = ['중요', '핵심', '비밀', '방법', '이유', '결국', '놀라', '절대', '첫 번째', '왜', '하지만', '사실']
    candidates = []
    for segment in segments:
        start = max(0, min(segment['start'], duration - length))
        end = min(duration, start + length)
        text = ' '.join(s['text'] for s in segments if s['end'] > start and s['start'] < end)
        score = sum(text.count(c) * 5 for c in cues) + len(text) / max(1, end - start)
        candidates.append((score, start, end, text))
    selected = []
    for score, start, end, text in sorted(candidates, reverse=True):
        if all(end <= item['start'] or start >= item['end'] for item in selected):
            selected.append(dict(start=start, end=end, title=text[:48].strip() or '핵심 장면', score=round(score, 1)))
        if len(selected) >= count:
            break
    return selected


def timestamp(seconds):
    ms = max(0, round(seconds * 1000))
    return f'{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}'


def subtitles(segments, start, end):
    rows = []
    for s in segments:
        a, b = max(start, s['start']), min(end, s['end'])
        if b > a:
            text = re.sub(r'<[^>]*>|\{[^}]*\}', '', s['text']).strip()
            chunks = [text[i:i + 22] for i in range(0, len(text), 22)]
            for i, chunk in enumerate(chunks):
                left = a - start + (b - a) * i / len(chunks)
                right = a - start + (b - a) * (i + 1) / len(chunks)
                rows.append(f'{len(rows)+1}\n{timestamp(left)} --> {timestamp(right)}\n{chunk}\n')
    return '\n'.join(rows)


def process(job_id, source, length, count):
    global model
    job = jobs[job_id]
    folder = source.parent
    try:
        job.update(stage='영상을 확인하고 있어요', progress=5)
        duration = probe(source)
        job.update(stage='한국어 음성을 분석하고 있어요 · 첫 실행은 모델 다운로드가 필요합니다', progress=15)
        with model_lock:
            if model is None:
                from faster_whisper import WhisperModel
                model = WhisperModel(os.getenv('WHISPER_MODEL', 'small'), device='cpu', compute_type='int8')
        stream, _ = model.transcribe(str(source), language='ko', vad_filter=True, word_timestamps=True)
        segments = []
        for s in stream:
            # Word-level timing keeps subtitles aligned when a clip begins inside a sentence.
            words = s.words or []
            # Group a few words so a subtitle remains readable rather than flashing per word.
            for offset in range(0, len(words), 5):
                group = words[offset:offset + 5]
                segments.append(dict(start=group[0].start, end=group[-1].end, text=''.join(w.word for w in group)))
            if not words:
                segments.append(dict(start=s.start, end=s.end, text=s.text))
            job['progress'] = min(60, 15 + int(s.end / duration * 45))
        (folder / 'transcript.json').write_text(json.dumps(segments, ensure_ascii=False), encoding='utf-8')
        clips = highlights(segments, duration, min(length, duration), count)
        if not segments:
            job['warning'] = '감지된 말소리가 없어 자동 자막과 대사 기반 추천을 생략했습니다. 첫 구간을 출력했습니다. 화면에 이미 들어 있는 글자는 음성 자막으로 추출하지 않습니다.'
            clips = [dict(start=0, end=min(length, duration), title='말소리 미감지 · 첫 구간', score=0)]
        job.update(stage='핵심 구간을 세로 영상으로 만들고 있어요', progress=65)
        for i, clip in enumerate(clips):
            srt = folder / f'clip-{i}.srt'
            srt.write_text(subtitles(segments, clip['start'], clip['end']), encoding='utf-8')
            output = folder / f'clip-{i}.mp4'
            filters = 'scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,setsar=1'
            if segments:
                filters += ",subtitles=" + str(srt) + ":force_style='FontName=Noto Sans CJK KR,FontSize=19,PrimaryColour=&H00FFFFFF,OutlineColour=&H00141414,BorderStyle=1,Outline=2,MarginV=36'"
            command(['ffmpeg', '-y', '-v', 'error', '-ss', str(clip['start']), '-i', str(source), '-t', str(clip['end'] - clip['start']), '-vf', filters, '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-threads', '2', '-c:a', 'aac', '-movflags', '+faststart', str(output)])
            clip.update(video=f'/api/jobs/{job_id}/files/clip-{i}.mp4', subtitle=f'/api/jobs/{job_id}/files/clip-{i}.srt' if segments else None)
            job['progress'] = 65 + int(30 * (i + 1) / len(clips))
        job.update(status='done', progress=100, stage='쇼츠가 완성됐어요', clips=clips)
    except Exception as exc:
        logging.exception('Video processing failed: %s', job_id)
        message = str(exc) if isinstance(exc, ValueError) else '처리에 실패했습니다. 모델 다운로드 연결과 FFmpeg·한국어 폰트 설치를 확인해 주세요.'
        job.update(status='error', stage=message)


@app.get('/')
def home():
    return FileResponse(ROOT / 'static/index.html')


@app.post('/api/jobs', status_code=202)
def create_job(file: UploadFile = File(...), length: int = Form(30), count: int = Form(3)):
    if length not in (15, 30, 60) or not 1 <= count <= 5:
        raise HTTPException(400, '길이는 15·30·60초, 개수는 1~5개를 선택해 주세요.')
    if sum(j['status'] == 'processing' for j in jobs.values()) >= 3:
        raise HTTPException(429, '작업이 많습니다. 진행 중인 작업이 끝난 뒤 시도해 주세요.')
    job_id = uuid.uuid4().hex
    folder = DATA / job_id
    folder.mkdir()
    source = folder / 'source'
    size = 0
    try:
        with source.open('wb') as out:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > 1024 ** 3:
                    raise HTTPException(413, '파일은 1GB 이하로 업로드해 주세요.')
                out.write(chunk)
        probe(source)
    except Exception as exc:
        shutil.rmtree(folder)
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(400, str(exc) if isinstance(exc, ValueError) else '올바른 영상 파일인지 확인해 주세요.')
    finally:
        file.file.close()
    jobs[job_id] = dict(id=job_id, status='processing', stage='분석 대기 중', progress=0, clips=[])
    pool.submit(process, job_id, source, length, count)
    return jobs[job_id]


@app.get('/api/jobs/{job_id}')
def get_job(job_id: str):
    return load_job(job_id)


@app.get('/api/jobs/{job_id}/files/{filename}')
def download(job_id: str, filename: str):
    if not re.fullmatch(r'[a-f0-9]{32}', job_id) or not re.fullmatch(r'clip-\d+\.(mp4|srt)|recreation\.(mp4|srt)|production\.zip', filename):
        raise HTTPException(404)
    job = load_job(job_id)
    if job.get('kind') == 'recreation' and (job['status'] != 'done' or not job.get('reviewed')):
        raise HTTPException(403, '완성본을 미리 보고 검수 항목을 확인한 뒤 다운로드해 주세요.')
    path = DATA / job_id / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, filename=filename)

# Recreation workflow: explicit source rights, original commentary and human review.
import zipfile
from pydantic import BaseModel, Field


class Brief(BaseModel):
    topic: str = Field(min_length=3, max_length=120)
    viewpoint: str = Field(min_length=10, max_length=500)
    facts: str = Field(min_length=10, max_length=3000)
    conclusion: str = Field(min_length=5, max_length=500)


@app.post('/api/recreation/draft')
def draft_script(brief: Brief):
    return {
        'script': f'[도입]\n{brief.topic}에 대해 살펴보겠습니다.\n\n[전개]\n{brief.facts.strip()}\n\n[나의 해설]\n{brief.viewpoint.strip()}\n\n[결론]\n{brief.conclusion.strip()}',
        'notice': '입력한 사실과 관점을 배치한 구성 초안입니다. 자동 사실 검증이나 생성형 AI 작성 결과가 아닙니다. 직접 수정한 뒤 녹음해 주세요.'
    }


def read_upload(upload, path, limit):
    size = 0
    try:
        with path.open('wb') as out:
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise ValueError('업로드 용량 제한을 초과했습니다.')
                out.write(chunk)
        if not size:
            raise ValueError('빈 파일은 사용할 수 없습니다.')
    finally:
        upload.file.close()


def audio_duration(path):
    info = json.loads(command(['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(path)]))
    if not any(s['codec_type'] == 'audio' for s in info['streams']):
        raise ValueError('직접 녹음한 음성 파일을 업로드해 주세요.')
    duration = float(info['format']['duration'])
    if not 6 <= duration <= 180:
        raise ValueError('내레이션 길이는 6초 이상 180초 이하로 준비해 주세요.')
    return duration


def recreation_video_duration(path):
    info = json.loads(command(['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(path)]))
    if not any(s['codec_type'] == 'video' for s in info['streams']):
        raise ValueError('원본 영상에 영상 트랙이 없습니다.')
    duration = float(info['format']['duration'])
    if not 1 <= duration <= 7200:
        raise ValueError('원본 영상은 1초 이상 2시간 이하로 준비해 주세요.')
    return duration


def save_job(job):
    (DATA / job['id'] / 'job.json').write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding='utf-8')


def load_job(job_id):
    if not re.fullmatch(r'[a-f0-9]{32}', job_id):
        raise HTTPException(404, '작업을 찾을 수 없습니다.')
    if job_id in jobs:
        return jobs[job_id]
    path = DATA / job_id / 'job.json'
    if path.is_file():
        job = json.loads(path.read_text(encoding='utf-8'))
        if job['status'] == 'processing':
            job.update(status='error', stage='서버 재시작으로 작업이 중단됐습니다. 다시 제작해 주세요.')
        jobs[job_id] = job
        return job
    raise HTTPException(404, '작업을 찾을 수 없습니다.')


def render_recreation(job_id):
    global model
    job = jobs[job_id]
    folder = DATA / job_id
    brief = job['brief']
    try:
        job.update(stage='직접 녹음한 내레이션을 분석하고 있어요', progress=10)
        save_job(job)
        with model_lock:
            if model is None:
                from faster_whisper import WhisperModel
                model = WhisperModel(os.getenv('WHISPER_MODEL', 'small'), device='cpu', compute_type='int8')
        stream, _ = model.transcribe(str(folder / 'narration'), language='ko', vad_filter=True, word_timestamps=True)
        segments = []
        for segment in stream:
            words = segment.words or []
            if words:
                for offset in range(0, len(words), 5):
                    group = words[offset:offset + 5]
                    segments.append(dict(start=group[0].start, end=group[-1].end, text='제작자 해설 · ' + ''.join(w.word for w in group)))
            else:
                segments.append(dict(start=segment.start, end=segment.end, text='제작자 해설 · ' + segment.text))
        if not segments:
            raise ValueError('내레이션에서 말소리를 찾지 못했습니다. 선명한 한국어 해설을 다시 녹음해 주세요.')
        duration = job['duration']
        srt = folder / 'recreation.srt'
        srt.write_text(subtitles(segments, 0, duration), encoding='utf-8')
        (folder / 'script.txt').write_text(brief['script'], encoding='utf-8')
        (folder / 'brief.json').write_text(json.dumps(brief, ensure_ascii=False, indent=2), encoding='utf-8')
        style = ":force_style='FontName=Noto Sans CJK KR,FontSize=19,PrimaryColour=&H00FFFFFF,OutlineColour=&H00141414,Outline=2,MarginV=45'"
        base = 'scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2:color=0x141719,setsar=1,fps=30,format=yuv420p'
        encoding = ['-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-threads', '2']
        parts = []
        def card(name, text, seconds):
            title = folder / (name + '.srt')
            # Strip subtitle control markup through the same caption writer.
            title.write_text(subtitles([dict(start=0, end=seconds, text=text)], 0, seconds), encoding='utf-8')
            target = folder / (name + '.mp4')
            card_style = ":force_style='FontName=Noto Sans CJK KR,FontSize=24,PrimaryColour=&H00D4F77D,Alignment=5,Outline=1,MarginL=55,MarginR=55'"
            command(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=0x141719:s=720x1280:r=30', '-t', str(seconds), '-vf', 'subtitles=' + str(title) + card_style, *encoding, str(target)])
            parts.append(target)
        card('intro', brief['topic'], 2)
        total = 2
        for index, scene in enumerate(brief['scenes']):
            target = folder / f'scene-{index}.mp4'
            seconds = scene['end'] - scene['start']
            command(['ffmpeg', '-y', '-v', 'error', '-ss', str(scene['start']), '-i', str(folder / 'source'), '-t', str(seconds), '-vf', base, *encoding, str(target)])
            parts.append(target)
            total += seconds
            job.update(stage='해설에 맞춰 장면을 구성하고 있어요', progress=35 + 25 * (index + 1) // len(brief['scenes']))
        card('outro', '제작자 해설 · ' + brief['conclusion'] + ' / 출처: ' + brief['credit'], duration - total)
        listing = folder / 'parts.txt'
        listing.write_text(''.join("file '" + str(p) + "'\n" for p in parts), encoding='utf-8')
        command(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', str(listing), '-c', 'copy', str(folder / 'visual.mp4')])
        job.update(stage='새 내레이션과 제작자 해설 자막을 합성하고 있어요', progress=80)
        command(['ffmpeg', '-y', '-v', 'error', '-i', str(folder / 'visual.mp4'), '-i', str(folder / 'narration'), '-map', '0:v:0', '-map', '1:a:0', '-vf', 'subtitles=' + str(srt) + style, '-t', str(duration), '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-threads', '2', '-c:a', 'aac', '-movflags', '+faststart', str(folder / 'recreation.mp4')])
        with zipfile.ZipFile(folder / 'production.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for filename in ['recreation.mp4', 'recreation.srt', 'script.txt', 'brief.json']:
                archive.write(folder / filename, filename)
            archive.write(ROOT / 'RECREATION_GUIDE.md', '재창작_지침.md')
        job.update(status='done', progress=100, stage='검수용 영상이 완성됐습니다. 미리 본 뒤 다운로드하세요.', preview=f'/api/recreation/{job_id}/preview')
        save_job(job)
    except Exception as exc:
        logging.exception('Recreation failed: %s', job_id)
        job.update(status='error', stage=str(exc) if isinstance(exc, ValueError) else '제작 중 오류가 발생했습니다. 음성 모델과 FFmpeg 설치를 확인해 주세요.')
        save_job(job)



@app.post('/api/recreation', status_code=202)
def create_recreation(file: UploadFile = File(...), narration: UploadFile = File(...), brief_json: str = Form(...)):
    folder = None
    try:
        brief = json.loads(brief_json)
        Brief(**brief)
        if brief.get('rights_confirmed') is not True or brief.get('script_confirmed') is not True:
            raise ValueError('소재 이용 권한과 독자적인 대본 작성을 확인해 주세요.')
        for key, limit in [('script', 6000), ('credit', 300), ('rights_basis', 1000)]:
            if not isinstance(brief.get(key), str) or not 5 <= len(brief[key].strip()) <= limit:
                raise ValueError('대본·출처·이용 권한 근거를 빠짐없이 입력해 주세요.')
        scenes = brief.get('scenes')
        if not isinstance(scenes, list) or not 1 <= len(scenes) <= 5:
            raise ValueError('설명에 필요한 장면을 1~5개 지정해 주세요.')
        if sum(j['status'] == 'processing' for j in jobs.values()) >= 3:
            raise HTTPException(429, '진행 중인 작업을 마친 뒤 다시 시도해 주세요.')
        job_id = uuid.uuid4().hex
        folder = DATA / job_id
        folder.mkdir()
        read_upload(file, folder / 'source', 1024 ** 3)
        read_upload(narration, folder / 'narration', 100 * 1024 ** 2)
        source_duration = recreation_video_duration(folder / 'source')
        duration = audio_duration(folder / 'narration')
        total = 0
        for scene in scenes:
            if not isinstance(scene, dict):
                raise ValueError('장면 형식이 올바르지 않습니다.')
            start, end = float(scene['start']), float(scene['end'])
            if not 0 <= start < end <= source_duration or not isinstance(scene.get('reason'), str) or not 5 <= len(scene['reason'].strip()) <= 500:
                raise ValueError('장면 시각은 영상 범위 안에 지정하고 사용 이유를 5자 이상 입력해 주세요.')
            scene.update(start=start, end=end)
            total += end - start
        if total + 4 > duration:
            raise ValueError('내레이션이 너무 짧습니다. 도입·마무리 4초와 선택한 장면 길이를 합친 시간 이상으로 녹음하거나 장면을 줄여 주세요.')
        jobs[job_id] = dict(id=job_id, kind='recreation', status='processing', stage='재창작 준비 중', progress=0, brief=brief, duration=duration, reviewed=False)
        save_job(jobs[job_id])
        pool.submit(render_recreation, job_id)
        return jobs[job_id]
    except HTTPException:
        if folder:
            shutil.rmtree(folder, ignore_errors=True)
        raise
    except (ValueError, KeyError, TypeError):
        if folder:
            shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(400, '기획·권리·대본·장면 시각·녹음 길이를 확인해 주세요. 내레이션은 선택 장면 총 길이 + 4초 이상, 180초 이하여야 합니다.')
    except Exception:
        if folder:
            shutil.rmtree(folder, ignore_errors=True)
        logging.exception('Recreation upload failed')
        raise HTTPException(400, '파일을 읽지 못했습니다. 올바른 영상·음성 파일인지 확인해 주세요.')
    finally:
        file.file.close()
        narration.file.close()


@app.get('/api/recreation/{job_id}/preview')
def recreation_preview(job_id: str):
    job = load_job(job_id)
    if job.get('kind') != 'recreation' or job['status'] != 'done':
        raise HTTPException(409, '아직 미리보기가 준비되지 않았습니다.')
    return FileResponse(DATA / job_id / 'recreation.mp4', media_type='video/mp4')


class Review(BaseModel):
    watched: bool
    rights: bool
    facts: bool
    music: bool
    privacy: bool
    original_value: bool


@app.post('/api/recreation/{job_id}/review')
def review_recreation(job_id: str, review: Review):
    job = load_job(job_id)
    if job.get('kind') != 'recreation' or job['status'] != 'done':
        raise HTTPException(409, '완성된 재창작 작업만 검수할 수 있습니다.')
    if not all(review.model_dump().values()):
        raise HTTPException(400, '미리보기와 모든 검수 항목을 확인해 주세요.')
    job['reviewed'] = True
    job['review'] = review.model_dump()
    job['downloads'] = {name: f'/api/jobs/{job_id}/files/{filename}' for name, filename in [('video', 'recreation.mp4'), ('subtitle', 'recreation.srt'), ('bundle', 'production.zip')]}
    save_job(job)
    with zipfile.ZipFile(DATA / job_id / 'production.zip', 'a', zipfile.ZIP_DEFLATED) as archive:
        if 'review.json' not in archive.namelist():
            archive.writestr('review.json', json.dumps(job['review'], ensure_ascii=False, indent=2))
    return job


@app.get('/api/downloads/guide')
def download_guide():
    return FileResponse(ROOT / 'RECREATION_GUIDE.md', filename='재창작_제작지침.md')


@app.get('/api/downloads/program')
def download_program():
    target = DATA / 'moment-recreation-program.zip'
    temp = DATA / f'program-{uuid.uuid4().hex}.zip'
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ['app.py', 'requirements.txt', 'requirements-dev.txt', 'README.md', 'RECREATION_GUIDE.md', '.gitignore', 'start.sh', 'start.bat']:
            path = ROOT / name
            if path.is_file():
                archive.write(path, f'moment-recreation/{name}')
        for directory in ['static', 'tests']:
            for path in (ROOT / directory).rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts:
                    archive.write(path, 'moment-recreation/' + str(path.relative_to(ROOT)))
    temp.replace(target)
    return FileResponse(target, filename='모먼트_재창작_프로그램.zip')
