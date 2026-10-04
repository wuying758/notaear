"""Prepare, serve, and audit a model-blind tonality annotation queue."""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import random
import subprocess
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .tonality import scale_pitch_classes
from .tonality_benchmark import _data_root, _load_source_abcs
from .validate_abc import validate_abc


EXACT_MODES = {
    "major", "natural_minor", "harmonic_minor", "melodic_minor",
    "dorian", "phrygian", "lydian", "mixolydian", "locrian",
    "gong", "shang", "jue", "zhi", "yu",
}
PARTIAL_MODES = {
    "minor_unspecified", "church_mode_unspecified",
    "pentatonic_unspecified", "tonal_mode_unspecified", "other",
}
MODES = EXACT_MODES | PARTIAL_MODES
JUDGEMENTS = {"tonal", "ambiguous", "skip"}
MODULATIONS = {"none", "present", "uncertain"}
UNCERTAINTY_REASONS = {
    "none", "too_short", "tonic_unclear", "mode_unclear",
    "possible_modulation", "broken_score", "other",
}


def _mode_compatible(human_mode: str, predicted_mode: str | None) -> bool:
    if human_mode == predicted_mode or human_mode == "tonal_mode_unspecified":
        return True
    families = {
        "minor_unspecified": {"natural_minor", "harmonic_minor", "melodic_minor"},
        "church_mode_unspecified": {
            "dorian", "phrygian", "lydian", "mixolydian", "locrian",
        },
        "pentatonic_unspecified": {"gong", "shang", "jue", "zhi", "yu"},
    }
    return predicted_mode in families.get(human_mode, set())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _item_key(identifier: str) -> str:
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:20]


def _easy_evidence(row: dict[str, Any], abc: str) -> dict[str, Any] | None:
    """Return objective readability evidence, without invoking the analyser."""
    from music21 import converter

    validation = validate_abc(abc)
    if not (validation.accepted and validation.rhythm_valid and validation.single_voice):
        return None
    score = converter.parseData(abc, format="abc")
    if len(score.parts) != 1:
        return None
    pitches = []
    for event in score.recurse().notes:
        if event.isChord:
            pitches.extend(pitch.pitchClass for pitch in event.pitches)
        else:
            pitches.append(event.pitch.pitchClass)
    if not pitches:
        return None
    measures = len(list(score.parts[0].getElementsByClass("Measure")))
    tonic = row["expected_tonic_pc"]
    scale = scale_pitch_classes(tonic, row["expected_mode"])
    coverage = sum(pitch in scale for pitch in pitches) / len(pitches)
    tonic_count = sum(pitch == tonic for pitch in pitches)
    if not (
        row["expected_mode"] in {"major", "natural_minor", "dorian"}
        and 32 <= len(pitches) <= 200
        and 8 <= measures <= 32
        and coverage >= 0.98
        and tonic_count >= 4
        and pitches[-1] == tonic
        and len(set(pitches)) >= 5
    ):
        return None
    penultimate_degree = (pitches[-2] - tonic) % 12 if len(pitches) > 1 else None
    # This ranking uses only declared-label musical evidence, never the model
    # under evaluation. It is an onboarding/calibration set, not an unbiased
    # estimate of real-world accuracy.
    clarity = (
        8.0 * coverage
        + 1.5 * min(tonic_count / len(pitches), 0.20) / 0.20
        + 0.8 * (penultimate_degree in {2, 7, 11})
        - 0.3 * abs(measures - 16) / 16
        - 0.2 * abs(len(pitches) - 96) / 96
    )
    return {
        "note_count": len(pitches),
        "measure_count": measures,
        "scale_coverage": round(coverage, 6),
        "tonic_count": tonic_count,
        "ends_on_declared_tonic": True,
        "penultimate_degree": penultimate_degree,
        "clarity_rank": round(clarity, 6),
        "strict_rhythm_valid": True,
        "single_voice_valid": True,
    }


def _render_preflight(abc: str) -> bool:
    """Require the exact anonymous score and MIDI paths used by the UI to work."""
    import verovio

    try:
        xml = _sanitized_musicxml(abc).decode("utf-8")
        toolkit = verovio.toolkit()
        toolkit.setOptions({
            "pageWidth": 2100, "pageHeight": 1200, "scale": 34,
            "adjustPageHeight": True, "breaks": "auto",
        })
        if not toolkit.loadData(xml) or toolkit.getPageCount() < 1:
            return False
        if not toolkit.renderToSVG(1):
            return False
        midi = base64.b64decode(toolkit.renderToMIDI())
        return len(midi) > 32
    except Exception:
        return False


def _select_easy_rows(
    rows: list[dict[str, Any]], manifest: Path, count: int,
) -> list[dict[str, Any]]:
    if count < 3:
        raise ValueError("easy profile requires at least 3 items")
    abcs = _load_source_abcs(rows, _data_root(manifest))
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        evidence = _easy_evidence(row, abcs[row["id"]])
        if evidence is not None and _render_preflight(abcs[row["id"]]):
            pools[row["expected_mode"]].append({**row, "selection_evidence": evidence})
    for pool in pools.values():
        pool.sort(key=lambda item: (
            -item["selection_evidence"]["clarity_rank"], item["id"]
        ))
    quotas = {"major": count // 3, "natural_minor": count // 3}
    quotas["dorian"] = count - sum(quotas.values())
    selected = []
    for mode in ("major", "natural_minor", "dorian"):
        if len(pools[mode]) < quotas[mode]:
            raise ValueError(
                f"easy profile has only {len(pools[mode])} eligible {mode} items; "
                f"{quotas[mode]} requested"
            )
        selected.extend(pools[mode][:quotas[mode]])
    # Alternate families instead of presenting one entire family in a block.
    selected.sort(key=lambda item: (
        max(0, ["major", "natural_minor", "dorian"].index(item["expected_mode"])),
        -item["selection_evidence"]["clarity_rank"],
    ))
    interleaved = []
    mode_lists = {
        mode: [row for row in selected if row["expected_mode"] == mode]
        for mode in ("major", "natural_minor", "dorian")
    }
    pattern = ("major", "natural_minor", "major", "natural_minor", "dorian")
    while len(interleaved) < len(selected):
        for mode in pattern:
            if mode_lists[mode]:
                interleaved.append(mode_lists[mode].pop(0))
    return interleaved


def prepare_queue(
    benchmark: Path,
    output: Path,
    *,
    repeat_fraction: float = 0.10,
    seed: int = 20260930,
    profile: str = "standard",
    manifest: Path | None = None,
    count: int = 24,
    namespace: str = "tonality-label-v1",
) -> dict[str, Any]:
    """Create a shuffled private queue with hidden duplicate audit items."""
    if not 0.0 <= repeat_fraction <= 1.0:
        raise ValueError("repeat_fraction must be between 0 and 1")
    rows = _read_jsonl(benchmark)
    if not rows:
        raise ValueError("benchmark is empty")
    identifiers = [row.get("id") for row in rows]
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ValueError("every benchmark row must have a non-empty id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("benchmark contains duplicate ids")
    if profile == "easy":
        if manifest is None:
            raise ValueError("easy profile requires a manifest")
        rows = _select_easy_rows(rows, manifest, count)
    elif profile != "standard":
        raise ValueError("profile must be standard or easy")

    rng = random.Random(seed)
    repeat_count = round(len(rows) * repeat_fraction)
    repeated = rng.sample(rows, repeat_count)
    occurrences: dict[str, int] = defaultdict(int)
    tasks = []
    for row in [*rows, *repeated]:
        identifier = row["id"]
        occurrence = occurrences[identifier]
        occurrences[identifier] += 1
        task_id = hashlib.sha256(
            f"{namespace}\0{seed}\0{identifier}\0{occurrence}".encode("utf-8")
        ).hexdigest()[:24]
        tasks.append({
            "task_id": task_id,
            "item_key": _item_key(identifier),
            "occurrence": occurrence,
            "locator": row,
        })
    rng.shuffle(tasks)
    queue = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": str(benchmark),
        "profile": profile,
        "namespace": namespace,
        "seed": seed,
        "item_count": len(rows),
        "repeat_count": repeat_count,
        "tasks": tasks,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return queue


def _public_task(task: dict[str, Any], index: int, total: int) -> dict[str, Any]:
    """Return only opaque routing data; no locator, label, or prediction."""
    return {"task_id": task["task_id"], "index": index, "total": total}


def _load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            task_id = json.loads(line)["task_id"]
            if isinstance(task_id, str):
                done.add(task_id)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return done


def _parse_byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Parse one RFC 7233 byte range, returning an inclusive interval."""
    if header is None:
        return None
    if size <= 0 or not header.startswith("bytes=") or "," in header:
        raise ValueError("unsupported byte range")
    value = header.removeprefix("bytes=").strip()
    if "-" not in value:
        raise ValueError("invalid byte range")
    start_text, end_text = value.split("-", 1)
    try:
        if not start_text:
            length = int(end_text)
            if length <= 0:
                raise ValueError
            start = max(0, size - length)
            end = size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
    except ValueError as error:
        raise ValueError("invalid byte range") from error
    if start < 0 or end < start or start >= size:
        raise ValueError("unsatisfiable byte range")
    return start, min(end, size - 1)


def _validate_annotation(payload: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    if payload.get("task_id") != task["task_id"]:
        raise ValueError("unknown task")
    judgement = payload.get("judgement")
    if judgement not in JUDGEMENTS:
        raise ValueError("invalid judgement")
    try:
        confidence = int(payload.get("confidence", 0))
    except (TypeError, ValueError) as error:
        raise ValueError("invalid confidence") from error
    if confidence not in (1, 2, 3):
        raise ValueError("confidence must be 1, 2, or 3")
    listened = payload.get("listened")
    if type(listened) is not bool:
        raise ValueError("listened must be boolean")
    notes = payload.get("notes", "")
    if not isinstance(notes, str) or len(notes) > 500:
        raise ValueError("notes must contain at most 500 characters")
    uncertainty_reason = payload.get("uncertainty_reason", "none")
    if uncertainty_reason not in UNCERTAINTY_REASONS:
        raise ValueError("invalid uncertainty reason")

    tonic_pc = payload.get("tonic_pc")
    mode = payload.get("mode")
    modulation = payload.get("modulation")
    if judgement == "tonal":
        if type(tonic_pc) is not int or not 0 <= tonic_pc <= 11:
            raise ValueError("tonal labels require tonic_pc from 0 to 11")
        if mode not in MODES:
            raise ValueError("tonal labels require a supported mode")
        if modulation not in MODULATIONS:
            raise ValueError("tonal labels require a modulation judgement")
    else:
        tonic_pc = None
        mode = None
        if judgement == "skip":
            modulation = None
        elif modulation not in MODULATIONS:
            raise ValueError("ambiguous labels require a modulation judgement")
    return {
        "judgement": judgement,
        "tonic_pc": tonic_pc,
        "mode": mode,
        "modulation": modulation,
        "confidence": confidence,
        "listened": listened,
        "uncertainty_reason": uncertainty_reason,
        "notes": notes.strip(),
    }


def _sanitized_musicxml(abc: str) -> bytes:
    """Render the pitches while removing textual and machine-readable answers."""
    from music21 import converter, expressions, harmony, key, stream
    from music21.musicxml.m21ToXml import GeneralObjectExporter

    parsed = converter.parseData(abc, format="abc")
    if len(parsed.parts) != 1:
        raise ValueError(f"expected one logical voice, found {len(parsed.parts)} parts")
    part = copy.deepcopy(parsed.parts[0])
    if not list(part.recurse().notes):
        raise ValueError("score contains no notes")
    part.id = "BlindPart"
    part.partName = None
    part.partAbbreviation = None
    for element in list(part.recurse()):
        if isinstance(element, (harmony.Harmony, expressions.TextExpression)):
            if element.activeSite is not None:
                element.activeSite.remove(element)
        elif isinstance(element, key.Key):
            # Keep the engraved key signature, but erase tonic/mode metadata.
            if element.activeSite is not None:
                element.activeSite.replace(element, key.KeySignature(element.sharps))
        if hasattr(element, "lyrics"):
            element.lyrics = []
    sanitized = stream.Score()
    sanitized.insert(0, part)
    xml = GeneralObjectExporter(sanitized).parse()
    lowered = xml.lower()
    if b"<mode>" in lowered or b"<harmony" in lowered:
        raise ValueError("sanitization left a tonality-bearing MusicXML field")
    return xml


def _html() -> str:
    return r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NotaEar 调性盲标</title><style>
:root{font-family:ui-sans-serif,system-ui,sans-serif;color:#172033;background:#f4f6fa}*{box-sizing:border-box}body{margin:0}.top{padding:14px 22px;background:#172033;color:white;display:flex;justify-content:space-between}.wrap{padding:18px;max-width:1450px;margin:auto}.card{background:white;border:1px solid #dce1ea;border-radius:12px;padding:16px;margin-bottom:14px}.score{min-height:300px;overflow:auto}.score svg{width:100%;height:auto}.play{display:flex;align-items:center;gap:12px;margin-bottom:10px}.play audio{display:none}.playbutton{width:44px;height:38px;border:0;border-radius:8px;background:#3567d6;color:white;cursor:pointer;font-size:16px}.timeline{flex:1;min-width:220px;accent-color:#3567d6;cursor:pointer}.time{font-variant-numeric:tabular-nums;min-width:92px;text-align:center;color:#4f5c70}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(125px,1fr));gap:8px}.choice input{position:absolute;opacity:0}.choice span{display:block;text-align:center;border:1px solid #bbc5d5;border-radius:8px;padding:10px 6px;cursor:pointer;background:white}.choice input:checked+span{background:#dce8ff;border-color:#3567d6;color:#183f91;font-weight:700}.section{margin:16px 0}.section h3{font-size:15px;margin:0 0 8px}.row{display:flex;gap:18px;align-items:center;flex-wrap:wrap}.submit{font-size:17px;padding:12px 24px;border:0;border-radius:9px;color:white;background:#3567d6;cursor:pointer}.secondary{background:#68758c}.danger{background:#95515b}.hint{font-size:13px;color:#6c778b}.error{color:#a82727;min-height:20px}.done{text-align:center;padding:70px;font-size:24px}.disabled{opacity:.55;pointer-events:none}textarea{width:100%;min-height:58px;border:1px solid #bbc5d5;border-radius:8px;padding:8px}
</style></head><body><div class="top"><strong>NotaEar · 调性/调式盲标</strong><span id="progress">载入中…</span></div><main class="wrap"><div id="work">
<section class="card"><div class="play"><button id="playbutton" class="playbutton" type="button" aria-label="播放">▶</button><input id="timeline" class="timeline" type="range" min="0" max="0" value="0" step="0.01" disabled aria-label="播放进度"><span id="time" class="time">0:00 / 0:00</span><audio id="audio" preload="metadata"></audio></div><div class="hint">可拖动时间轴定位；系统会记录你是否实际播放过，供分层评估。调号保留，但调名、来源、弱标签和模型结果均隐藏。</div><div id="score" class="score"></div></section>
<section class="card"><div class="section"><h3>1. 主音</h3><div class="grid" id="tonics"></div></div><div class="section"><h3>2. 调式（证据不够时选大类，不要强猜具体型）</h3><div class="grid" id="modes"></div></div>
<div class="section"><h3>3. 是否存在中途转调</h3><div class="row"><label><input type="radio" name="modulation" value="none" checked> 无</label><label><input type="radio" name="modulation" value="present"> 有</label><label><input type="radio" name="modulation" value="uncertain"> 不确定</label></div></div>
<div class="section"><h3>4. 把握</h3><div class="row"><label><input type="radio" name="confidence" value="1"> 低</label><label><input type="radio" name="confidence" value="2" checked> 中</label><label><input type="radio" name="confidence" value="3"> 高</label></div></div>
<div class="section"><h3>不确定或跳过的主要原因</h3><select id="reason"><option value="none">无</option><option value="too_short">片段太短 / 信息不足</option><option value="tonic_unclear">主音不明确</option><option value="mode_unclear">调式具体型不明确</option><option value="possible_modulation">疑似转调或调式交替</option><option value="broken_score">谱面或播放有错误</option><option value="other">其他</option></select></div>
<div class="section"><h3>备注（可选）</h3><textarea id="notes" maxlength="500" placeholder="例如：终止音不明确、疑似调式交替……"></textarea></div><div id="error" class="error"></div>
<div class="row"><button class="submit" onclick="submitLabel('tonal')">保存调性判断</button><button class="submit secondary" onclick="submitLabel('ambiguous')">无法唯一判断</button><button class="submit danger" onclick="submitLabel('skip')">谱面错误 / 跳过</button></div></section></div><div id="done" class="done" hidden></div></main>
<script>
const tonicNames=['C','C♯ / D♭','D','D♯ / E♭','E','F','F♯ / G♭','G','G♯ / A♭','A','A♯ / B♭','B'];
const modeNames={major:'大调',minor_unspecified:'小调（种类不明）',natural_minor:'自然小调',harmonic_minor:'和声小调',melodic_minor:'旋律小调',church_mode_unspecified:'中古调式（具体型不明）',dorian:'Dorian',phrygian:'Phrygian',lydian:'Lydian',mixolydian:'Mixolydian',locrian:'Locrian',pentatonic_unspecified:'五声调式（具体型不明）',gong:'宫调式',shang:'商调式',jue:'角调式',zhi:'徵调式',yu:'羽调式',tonal_mode_unspecified:'有主音但调式不明',other:'其他'};
function choices(root,name,values){root.innerHTML=values.map(([value,label])=>`<label class="choice"><input type="radio" name="${name}" value="${value}"><span>${label}</span></label>`).join('')}
choices(document.querySelector('#tonics'),'tonic',tonicNames.map((x,i)=>[i,x]));choices(document.querySelector('#modes'),'mode',Object.entries(modeNames));
let task=null,busy=false,listened=false;const audio=document.querySelector('#audio'),timeline=document.querySelector('#timeline'),playbutton=document.querySelector('#playbutton'),timeLabel=document.querySelector('#time');
function formatTime(value){if(!Number.isFinite(value)||value<0)return '0:00';let whole=Math.floor(value),minutes=Math.floor(whole/60),seconds=String(whole%60).padStart(2,'0');return `${minutes}:${seconds}`}
function updateTime(){let current=Number(timeline.value)||0,duration=Number.isFinite(audio.duration)?audio.duration:0;timeLabel.textContent=`${formatTime(current)} / ${formatTime(duration)}`}
function syncDuration(){let duration=Number.isFinite(audio.duration)?audio.duration:0;timeline.max=String(duration);timeline.disabled=duration<=0;if(Number(timeline.value)>duration)timeline.value=String(duration);updateTime()}
audio.addEventListener('loadedmetadata',syncDuration);audio.addEventListener('durationchange',syncDuration);audio.addEventListener('timeupdate',()=>{if(!timeline.matches(':active'))timeline.value=String(audio.currentTime);updateTime()});audio.addEventListener('play',()=>{listened=true;playbutton.textContent='❚❚';playbutton.setAttribute('aria-label','暂停')});audio.addEventListener('pause',()=>{playbutton.textContent='▶';playbutton.setAttribute('aria-label','播放')});audio.addEventListener('ended',()=>{playbutton.textContent='▶'});playbutton.addEventListener('click',async()=>{if(audio.paused){try{await audio.play()}catch(error){document.querySelector('#error').textContent='音频暂时无法播放，请重试。'}}else audio.pause()});timeline.addEventListener('input',()=>{if(Number.isFinite(audio.duration)){audio.currentTime=Number(timeline.value);updateTime()}});
async function loadNext(){audio.pause();audio.removeAttribute('src');audio.load();timeline.value='0';timeline.max='0';timeline.disabled=true;playbutton.textContent='▶';updateTime();listened=false;document.querySelector('#error').textContent='';document.querySelector('#notes').value='';document.querySelector('#reason').value='none';document.querySelectorAll('input[name=tonic],input[name=mode]').forEach(x=>x.checked=false);let r=await fetch('/api/next');let s=await r.json();if(s.done){document.querySelector('#work').hidden=true;let d=document.querySelector('#done');d.hidden=false;d.textContent=`本轮 ${s.completed}/${s.total} 条已完成。请运行 audit 查看重复样本一致性。`;document.querySelector('#progress').textContent='完成';return}task=s.task;document.querySelector('#progress').textContent=`${s.completed}/${s.total} 已标`;audio.src=`/audio/${task.task_id}.wav`;audio.load();let el=document.querySelector('#score');el.innerHTML='<span class="hint">正在排版…</span>';let score=await fetch(`/score/${task.task_id}.svg`);if(!score.ok){el.textContent='谱面渲染失败；请使用“谱面错误 / 跳过”。';return}el.innerHTML=await score.text()}
async function submitLabel(judgement){if(!task||busy)return;let tonic=document.querySelector('input[name=tonic]:checked');let mode=document.querySelector('input[name=mode]:checked');if(judgement==='tonal'&&(!tonic||!mode)){document.querySelector('#error').textContent='保存调性判断前必须选择主音和调式；具体型不清楚时可选相应大类。';return}busy=true;document.querySelector('#work').classList.add('disabled');let payload={task_id:task.task_id,judgement,tonic_pc:tonic?Number(tonic.value):null,mode:mode?mode.value:null,modulation:document.querySelector('input[name=modulation]:checked').value,confidence:Number(document.querySelector('input[name=confidence]:checked').value),listened,uncertainty_reason:document.querySelector('#reason').value,notes:document.querySelector('#notes').value};let r=await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});if(!r.ok){document.querySelector('#error').textContent=(await r.json()).error||'保存失败'}else await loadNext();busy=false;document.querySelector('#work').classList.remove('disabled')}
loadNext();
</script></body></html>'''


class TonalityLabelServer(ThreadingHTTPServer):
    def __init__(
        self,
        address,
        queue: dict[str, Any],
        manifest: Path,
        annotation_path: Path,
        annotator: str,
        audio_cache: Path,
        soundfont: Path,
    ):
        import verovio

        super().__init__(address, TonalityLabelHandler)
        self.queue = queue
        self.annotation_path = annotation_path
        self.annotator = annotator
        self.soundfont = soundfont
        self.lock = threading.Lock()
        self.render_lock = threading.Lock()
        self.audio_lock = threading.Lock()
        self.audio_cache = audio_cache
        self.audio_cache.mkdir(parents=True, exist_ok=True)
        self.renderer = verovio.toolkit()
        self.renderer.setOptions({
            "pageWidth": 2100, "pageHeight": 1200, "scale": 34,
            "adjustPageHeight": True, "breaks": "auto",
        })
        self.tasks = {task["task_id"]: task for task in queue["tasks"]}
        unique = {task["locator"]["id"]: task["locator"] for task in queue["tasks"]}
        self.abcs = _load_source_abcs(unique.values(), _data_root(manifest))
        missing = set(unique) - set(self.abcs)
        if missing:
            raise ValueError(f"could not load {len(missing)} queue scores")
        self.xml_cache: dict[str, bytes] = {}

    def musicxml(self, task: dict[str, Any]) -> bytes:
        item_key = task["item_key"]
        with self.render_lock:
            if item_key not in self.xml_cache:
                self.xml_cache[item_key] = _sanitized_musicxml(
                    self.abcs[task["locator"]["id"]]
                )
            return self.xml_cache[item_key]


class TonalityLabelHandler(BaseHTTPRequestHandler):
    server: TonalityLabelServer

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def _json(self, payload, status=HTTPStatus.OK):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _task_from_path(self, prefix: str, suffix: str):
        path = urlparse(self.path).path
        if not path.startswith(prefix) or not path.endswith(suffix):
            return None
        return self.server.tasks.get(path.removeprefix(prefix).removesuffix(suffix))

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            data = _html().encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/api/next":
            done = _load_done(self.server.annotation_path)
            tasks = self.server.queue["tasks"]
            pending = [(index, task) for index, task in enumerate(tasks)
                       if task["task_id"] not in done]
            completed = len(tasks) - len(pending)
            if not pending:
                self._json({"done": True, "completed": completed, "total": len(tasks)})
            else:
                index, task = pending[0]
                self._json({"done": False, "completed": completed, "total": len(tasks),
                            "task": _public_task(task, index, len(tasks))})
            return
        task = self._task_from_path("/score/", ".svg")
        if task is not None:
            try:
                xml = self.server.musicxml(task).decode("utf-8")
                with self.server.render_lock:
                    if not self.server.renderer.loadData(xml):
                        raise ValueError("Verovio could not load score")
                    data = "\n".join(
                        self.server.renderer.renderToSVG(page)
                        for page in range(1, self.server.renderer.getPageCount() + 1)
                    ).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "private, max-age=86400")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as error:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
            return
        task = self._task_from_path("/audio/", ".wav")
        if task is not None:
            wav_path = self.server.audio_cache / f"{task['item_key']}.wav"
            try:
                with self.server.audio_lock:
                    if not wav_path.exists():
                        xml = self.server.musicxml(task).decode("utf-8")
                        with self.server.render_lock:
                            if not self.server.renderer.loadData(xml):
                                raise ValueError("Verovio could not load score")
                            midi = base64.b64decode(self.server.renderer.renderToMIDI())
                        token = uuid.uuid4().hex
                        midi_path = self.server.audio_cache / f".{task['item_key']}-{token}.mid"
                        temp_wav = self.server.audio_cache / f".{task['item_key']}-{token}.wav"
                        try:
                            midi_path.write_bytes(midi)
                            result = subprocess.run([
                                "fluidsynth", "-ni", "-g", "0.7", str(self.server.soundfont),
                                str(midi_path), "-F", str(temp_wav), "-r", "22050",
                            ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               text=True, timeout=60, check=False)
                            if result.returncode or not temp_wav.exists():
                                raise RuntimeError(result.stderr.strip() or "FluidSynth failed")
                            temp_wav.replace(wav_path)
                        finally:
                            midi_path.unlink(missing_ok=True)
                            temp_wav.unlink(missing_ok=True)
                data = wav_path.read_bytes()
                try:
                    byte_range = _parse_byte_range(self.headers.get("Range"), len(data))
                except ValueError:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{len(data)}")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if byte_range is None:
                    status = HTTPStatus.OK
                    body = data
                else:
                    start, end = byte_range
                    status = HTTPStatus.PARTIAL_CONTENT
                    body = data[start:end + 1]
                self.send_response(status)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Cache-Control", "private, max-age=86400")
                self.send_header("Accept-Ranges", "bytes")
                if byte_range is not None:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as error:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if urlparse(self.path).path != "/api/label":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 4096:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            task = self.server.tasks.get(payload.get("task_id"))
            if task is None:
                raise ValueError("unknown task")
            human = _validate_annotation(payload, task)
            with self.server.lock:
                if task["task_id"] in _load_done(self.server.annotation_path):
                    self._json({"error": "task already labeled"}, HTTPStatus.CONFLICT)
                    return
                locator = task["locator"]
                record = {
                    "schema_version": 2,
                    "annotation_id": str(uuid.uuid4()),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "annotator": self.server.annotator,
                    "task_id": task["task_id"],
                    "item_key": task["item_key"],
                    "occurrence": task["occurrence"],
                    "identity": {
                        key: locator[key] for key in
                        ("id", "score_id", "source", "work_fingerprint", "target_key",
                         "shard", "line_number") if key in locator
                    },
                    "human": human,
                    "weak_reference": {
                        "kind": locator.get("label_kind"),
                        "tonic_pc": locator.get("expected_tonic_pc"),
                        "mode": locator.get("expected_mode"),
                    },
                }
                self.server.annotation_path.parent.mkdir(parents=True, exist_ok=True)
                with self.server.annotation_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._json({"saved": True})
        except (ValueError, json.JSONDecodeError, TypeError) as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)


def audit(queue_path: Path, annotations: Path) -> dict[str, Any]:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    records = _read_jsonl(annotations) if annotations.exists() else []
    valid_tasks = {task["task_id"] for task in queue["tasks"]}
    records = [record for record in records if record.get("task_id") in valid_tasks]
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_item[record["item_key"]].append(record)
    repeated = [values for values in by_item.values() if len(values) >= 2]

    def label(record):
        human = record["human"]
        return (human["judgement"], human.get("tonic_pc"), human.get("mode"))

    def agreement(field):
        comparable = 0
        equal = 0
        for values in repeated:
            first, second = values[:2]
            left = field(first)
            right = field(second)
            comparable += 1
            equal += left == right
        return round(equal / comparable, 6) if comparable else None

    unique = {}
    for record in records:
        unique.setdefault(record["item_key"], record)
    tonal = [record for record in unique.values()
             if record.get("human", {}).get("judgement") == "tonal"]
    tonic_matches = sum(record["human"]["tonic_pc"] == record["weak_reference"]["tonic_pc"]
                        for record in tonal)
    exact_mode_items = [record for record in tonal
                        if record["human"]["mode"] in EXACT_MODES]
    mode_matches = sum(record["human"]["mode"] == record["weak_reference"]["mode"]
                       for record in exact_mode_items)
    joint_matches = sum(
        record["human"]["tonic_pc"] == record["weak_reference"]["tonic_pc"]
        and record["human"]["mode"] == record["weak_reference"]["mode"]
        for record in tonal
    )
    compatible_matches = sum(
        _mode_compatible(record["human"]["mode"], record["weak_reference"]["mode"])
        for record in tonal
    )
    ratio = lambda value, total: round(value / total, 6) if total else None
    return {
        "schema_version": 1,
        "queue": str(queue_path),
        "annotations": str(annotations),
        "task_completion": {"completed": len({r["task_id"] for r in records}),
                            "total": len(queue["tasks"])},
        "unique_items_labeled": len(unique),
        "judgement_counts": dict(sorted(
            (name, sum(r["human"]["judgement"] == name for r in unique.values()))
            for name in JUDGEMENTS
        )),
        "repeat_consistency": {
            "completed_pairs": len(repeated),
            "combined_label_agreement": agreement(label),
            "modulation_agreement": agreement(lambda r: r["human"].get("modulation")),
            "listening_agreement": agreement(lambda r: r["human"].get("listened")),
        },
        "human_vs_declared_weak_reference": {
            "tonal_items": len(tonal),
            "tonic_agreement": ratio(tonic_matches, len(tonal)),
            "exact_mode_items": len(exact_mode_items),
            "exact_mode_agreement": ratio(mode_matches, len(exact_mode_items)),
            "mode_compatible_agreement": ratio(compatible_matches, len(tonal)),
            "joint_agreement": ratio(joint_matches, len(tonal)),
            "warning": "This measures agreement with ABC K: metadata, not human accuracy.",
        },
    }


def evaluate_human(
    queue_path: Path, annotations: Path, manifest: Path,
) -> dict[str, Any]:
    """Evaluate hidden-key inference against unique human-labelled real scores."""
    from collections import Counter

    from .difficulty import analyze_abc

    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    tasks = {task["task_id"]: task for task in queue["tasks"]}
    records = [record for record in _read_jsonl(annotations)
               if record.get("task_id") in tasks]
    completed = len({record["task_id"] for record in records})
    if completed != len(tasks):
        raise ValueError(
            f"annotation queue is incomplete: {completed}/{len(tasks)} tasks"
        )
    unique: dict[str, dict[str, Any]] = {}
    for record in sorted(records, key=lambda item: item.get("occurrence", 0)):
        unique.setdefault(record["item_key"], record)
    tonal = [record for record in unique.values()
             if record["human"]["judgement"] == "tonal"]
    locator_rows = []
    for record in tonal:
        identity = record["identity"]
        locator_rows.append({
            "id": identity["id"],
            "target_key": identity["target_key"],
            "shard": identity["shard"],
            "line_number": identity["line_number"],
        })
    abcs = _load_source_abcs(locator_rows, _data_root(manifest))
    counts = Counter()
    statuses = Counter()
    per_mode: dict[str, Counter] = defaultdict(Counter)
    confusions = Counter()
    failures = []
    accepted_statuses = {"confident", "declared"}
    for record in tonal:
        human = record["human"]
        identifier = record["identity"]["id"]
        try:
            result = analyze_abc(
                abcs[identifier], use_declared_tonality=False
            ).tonality
        except Exception as error:
            failures.append({"id": identifier, "error": f"{type(error).__name__}: {error}"})
            continue
        primary = result["primary"]
        predicted_tonic = primary["tonic_pc"] if primary else None
        predicted_mode = primary["mode"] if primary else None
        tonic_correct = predicted_tonic == human["tonic_pc"]
        exact_mode = human["mode"] in EXACT_MODES
        exact_mode_correct = exact_mode and predicted_mode == human["mode"]
        compatible_mode = _mode_compatible(human["mode"], predicted_mode)
        joint = tonic_correct and compatible_mode
        accepted = result["status"] in accepted_statuses
        counts.update(
            total=1,
            tonic_correct=tonic_correct,
            exact_mode_items=exact_mode,
            exact_mode_correct=exact_mode_correct,
            compatible_mode_correct=compatible_mode,
            joint_correct=joint,
            accepted=accepted,
            accepted_joint_correct=accepted and joint,
        )
        statuses[result["status"]] += 1
        per_mode[human["mode"]].update(total=1, joint_correct=joint)
        confusions[(human["mode"], predicted_mode or "__abstain__")] += 1

    def ratio(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 6) if denominator else None

    total = counts["total"]
    accepted = counts["accepted"]
    return {
        "schema_version": 1,
        "queue": str(queue_path),
        "annotations": str(annotations),
        "manifest": str(manifest),
        "label_kind": "single_annotator_blind_human",
        "unique_items": len(unique),
        "tonal_items": len(tonal),
        "skipped_or_ambiguous_items": len(unique) - len(tonal),
        "evaluated_items": total,
        "failures": failures,
        "tonic_accuracy": ratio(counts["tonic_correct"], total),
        "exact_mode_items": counts["exact_mode_items"],
        "exact_mode_accuracy": ratio(
            counts["exact_mode_correct"], counts["exact_mode_items"]
        ),
        "compatible_mode_accuracy": ratio(counts["compatible_mode_correct"], total),
        "joint_compatible_accuracy": ratio(counts["joint_correct"], total),
        "accepted_coverage": ratio(accepted, total),
        "accepted_joint_accuracy": ratio(counts["accepted_joint_correct"], accepted),
        "status_counts": dict(sorted(statuses.items())),
        "per_human_mode": {
            mode: {
                "items": values["total"],
                "joint_compatible_accuracy": ratio(
                    values["joint_correct"], values["total"]
                ),
            }
            for mode, values in sorted(per_mode.items())
        },
        "confusions": [
            {"human": human, "predicted": predicted, "count": count}
            for (human, predicted), count in sorted(confusions.items())
            if human != predicted
        ],
        "limitations": [
            f"Only one annotator and {len(tonal)} usable tonal items are available.",
            "The easy profile intentionally selects clear cadences and is not representative.",
            "Human exact mode and modulation consistency require a second annotator.",
            "Heuristic confidence is not calibrated as a probability.",
        ],
    }


def serve(
    queue_path: Path,
    manifest: Path,
    annotations: Path,
    host: str,
    port: int,
    annotator: str,
    soundfont: Path,
) -> None:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    server = TonalityLabelServer(
        (host, port), queue, manifest, annotations, annotator,
        queue_path.parent / "audio", soundfont,
    )
    print(f"NotaEar blind tonality labeler: http://{host}:{port}")
    print(f"Annotations: {annotations}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="build a shuffled blind queue")
    prepare.add_argument("--benchmark", type=Path,
                         default=Path("data/evaluation/tonality/declared-v1.jsonl"))
    prepare.add_argument("--output", type=Path,
                         default=Path("out/tonality-label-v1/queue.json"))
    prepare.add_argument("--repeat-fraction", type=float, default=0.10)
    prepare.add_argument("--seed", type=int, default=20260930)
    prepare.add_argument("--profile", choices=("standard", "easy"), default="standard")
    prepare.add_argument("--manifest", type=Path,
                         default=Path("../data/notaear-manifests/v-yidiao-v1/manifest.jsonl"))
    prepare.add_argument("--count", type=int, default=24,
                         help="unique item count for the easy profile")
    prepare.add_argument("--namespace", default="tonality-label-v1",
                         help="task identity namespace; change for a fresh labeling round")
    run = commands.add_parser("serve", help="run the local blind-labeling UI")
    run.add_argument("--queue", type=Path,
                     default=Path("out/tonality-label-v1/queue.json"))
    run.add_argument("--manifest", type=Path,
                     default=Path("../data/notaear-manifests/v-yidiao-v1/manifest.jsonl"))
    run.add_argument("--annotations", type=Path,
                     default=Path("data/annotations/tonality_gold_v1.jsonl"))
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8766)
    run.add_argument("--annotator", default="Daniel Wu")
    run.add_argument("--soundfont", type=Path,
                     default=Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"))
    report = commands.add_parser("audit", help="report completion and repeat consistency")
    report.add_argument("--queue", type=Path,
                        default=Path("out/tonality-label-v1/queue.json"))
    report.add_argument("--annotations", type=Path,
                        default=Path("data/annotations/tonality_gold_v1.jsonl"))
    report.add_argument("--output", type=Path)
    evaluate = commands.add_parser(
        "evaluate-human", help="evaluate hidden-key inference against human labels"
    )
    evaluate.add_argument("--queue", type=Path,
                          default=Path("out/tonality-label-easy-v3/queue.json"))
    evaluate.add_argument("--annotations", type=Path,
                          default=Path("data/annotations/tonality_gold_v2.jsonl"))
    evaluate.add_argument("--manifest", type=Path,
                          default=Path("../data/notaear-manifests/v-yidiao-v1/manifest.jsonl"))
    evaluate.add_argument("--output", type=Path,
                          default=Path("out/tonality-label-easy-v3/human-evaluation.json"))
    args = parser.parse_args(argv)
    if args.command == "prepare":
        queue = prepare_queue(
            args.benchmark, args.output,
            repeat_fraction=args.repeat_fraction, seed=args.seed,
            profile=args.profile, manifest=args.manifest, count=args.count,
            namespace=args.namespace,
        )
        result = {"output": str(args.output), "unique_items": queue["item_count"],
                  "repeat_items": queue["repeat_count"], "tasks": len(queue["tasks"])}
    elif args.command == "serve":
        serve(args.queue, args.manifest, args.annotations, args.host, args.port,
              args.annotator, args.soundfont)
        return 0
    elif args.command == "audit":
        result = audit(args.queue, args.annotations)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    else:
        result = evaluate_human(args.queue, args.annotations, args.manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
