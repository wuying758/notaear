"""Local, model-blind pairwise labeling for sight-singing difficulty."""
from __future__ import annotations

import argparse
import hashlib
import base64
import json
import math
import random
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from zipfile import ZipFile

from .train_difficulty import (
    FEATURE_NAMES,
    FILENAME_RE,
    STAGE_SERIES,
    extract_musicxml_features,
    score_dimensions,
)


JUDGEMENTS = {"left_harder", "equal", "right_harder", "skip"}
DIMENSIONS = {"overall", "pitch", "rhythm", "tonality", "notation", "length"}


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    source: str
    archive: str
    member: str
    sha256: str
    features: dict[str, float]
    expected_series: float
    uncertainty: float
    predicted_series: int
    dimension_percentiles: dict[str, float]


def _candidate_id(source: str, member: str, digest: str) -> str:
    return hashlib.sha256(f"{source}\0{member}\0{digest}".encode()).hexdigest()[:20]


def _predict_many(feature_rows, bundle):
    import numpy as np

    matrix = np.asarray([[features[name] for name in bundle["feature_names"]]
                         for features in feature_rows], dtype=np.float32)
    raw = bundle["model"].predict_proba(matrix)
    probabilities = np.zeros((len(feature_rows), len(bundle["stage_series"])), dtype=np.float64)
    for column, class_id in enumerate(bundle["model"].classes_):
        probabilities[:, int(class_id)] = raw[:, column]
    expected = 1.0 + 2.0 * (probabilities @ np.arange(probabilities.shape[1]))
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=1)
    entropy /= math.log(probabilities.shape[1])
    predicted = [int(bundle["stage_series"][index]) for index in probabilities.argmax(axis=1)]
    return expected, entropy, predicted


def _load_candidates(source: str, archive_path: Path, bundle) -> list[Candidate]:
    raw_candidates = []
    with ZipFile(archive_path) as archive:
        for member in sorted(archive.namelist()):
            if not member.lower().endswith((".xml", ".musicxml")):
                continue
            if source == "melodia":
                match = FILENAME_RE.search(member)
                if not match or int(match.group(2)) not in STAGE_SERIES:
                    continue
            xml = archive.read(member)
            try:
                features, part_count, _ = extract_musicxml_features(xml)
            except (ValueError, KeyError):
                continue
            if part_count != 1 or not xml.count(b"<pitch>"):
                continue
            digest = hashlib.sha256(xml).hexdigest()
            raw_candidates.append((member, digest, features))
    predictions = _predict_many([item[2] for item in raw_candidates], bundle)
    candidates = []
    for (member, digest, features), expected, uncertainty, predicted in zip(*[raw_candidates, *predictions]):
        candidates.append(Candidate(
            candidate_id=_candidate_id(source, member, digest),
            source=source,
            archive=str(archive_path),
            member=member,
            sha256=digest,
            features=features,
            expected_series=round(float(expected), 6),
            uncertainty=round(float(uncertainty), 6),
            predicted_series=predicted,
            dimension_percentiles=score_dimensions(features, bundle["calibration"]),
        ))
    return candidates


def _dimension_distance(left: Candidate, right: Candidate) -> float:
    keys = left.dimension_percentiles.keys()
    return sum(abs(left.dimension_percentiles[key] - right.dimension_percentiles[key]) for key in keys) / len(keys)


def _select_pairs(candidates: list[Candidate], count: int, seed: int) -> list[dict]:
    """Choose close-score, uncertain, feature-diverse comparisons."""
    rng = random.Random(seed)
    by_source = {
        source: [candidate for candidate in candidates if candidate.source == source]
        for source in {candidate.source for candidate in candidates}
    }
    eyes = sorted(by_source.get("eyes_and_ears", []), key=lambda item: item.uncertainty, reverse=True)
    melodia = sorted(by_source.get("melodia", []), key=lambda item: item.uncertainty, reverse=True)
    if not eyes or not melodia:
        raise ValueError("both Eyes and Ears and Melodia candidates are required")

    pairs, used_pairs = [], set()
    usage: dict[str, int] = {}
    modes = ("eyes-eyes", "eyes-melodia", "eyes-eyes", "melodia-melodia")
    anchors = eyes[:max(100, count * 3)]
    rng.shuffle(anchors)
    attempts = 0
    while len(pairs) < count and attempts < count * 50:
        mode = modes[len(pairs) % len(modes)]
        if mode == "eyes-eyes":
            left_pool, right_pool = anchors, eyes
        elif mode == "eyes-melodia":
            left_pool, right_pool = anchors, melodia
        else:
            left_pool, right_pool = melodia[:max(100, count * 3)], melodia
        anchor = left_pool[attempts % len(left_pool)]
        eligible = [item for item in right_pool
                    if item.candidate_id != anchor.candidate_id
                    and usage.get(item.candidate_id, 0) < 2]
        if not eligible:
            attempts += 1
            continue
        partner = max(eligible, key=lambda item: (
            item.uncertainty + anchor.uncertainty
            + min(_dimension_distance(anchor, item), 60.0) / 80.0
            - abs(anchor.expected_series - item.expected_series) * 1.5
            - usage.get(item.candidate_id, 0) * 0.4
        ))
        pair_key = tuple(sorted((anchor.candidate_id, partner.candidate_id)))
        attempts += 1
        if pair_key in used_pairs or abs(anchor.expected_series - partner.expected_series) > 1.4:
            continue
        left, right = (anchor, partner) if rng.random() < 0.5 else (partner, anchor)
        pair_id = hashlib.sha256(f"{seed}:{pair_key[0]}:{pair_key[1]}".encode()).hexdigest()[:20]
        pairs.append({"pair_id": pair_id, "left": asdict(left), "right": asdict(right)})
        used_pairs.add(pair_key)
        usage[anchor.candidate_id] = usage.get(anchor.candidate_id, 0) + 1
        usage[partner.candidate_id] = usage.get(partner.candidate_id, 0) + 1
    if len(pairs) < count:
        raise RuntimeError(f"could only select {len(pairs)} of {count} requested pairs")
    return pairs


def prepare_queue(model_path: Path, melodia: Path, eyes: Path, output: Path,
                  count: int = 60, seed: int = 20260928) -> dict:
    import joblib

    bundle = joblib.load(model_path)
    candidates = (_load_candidates("melodia", melodia, bundle)
                  + _load_candidates("eyes_and_ears", eyes, bundle))
    queue = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model_path.parent.name,
        "seed": seed,
        "candidate_count": len(candidates),
        "pairs": _select_pairs(candidates, count, seed),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")
    return queue


def _public_pair(pair: dict, index: int, total: int) -> dict:
    def clean(side):
        return {"candidate_id": side["candidate_id"]}
    return {"pair_id": pair["pair_id"], "index": index, "total": total,
            "left": clean(pair["left"]), "right": clean(pair["right"])}


def _load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            done.add(json.loads(line)["pair_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def _html() -> str:
    return r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NotaEar 难度比较</title>
<style>
:root{font-family:ui-sans-serif,system-ui,sans-serif;color:#172033;background:#f4f6fa}*{box-sizing:border-box}
body{margin:0}.top{padding:14px 22px;background:#172033;color:white;display:flex;justify-content:space-between}.wrap{padding:18px;max-width:1800px;margin:auto}
.scores{display:grid;grid-template-columns:1fr 1fr;gap:14px}.card{background:white;border:1px solid #dce1ea;border-radius:12px;padding:10px;min-height:390px;overflow:auto}.cardhead{display:flex;align-items:center;justify-content:center;gap:18px;position:sticky;left:0}.card h2{text-align:center;margin:4px}.card audio{height:34px;width:230px}.score{min-width:650px}.score svg{width:100%;height:auto}
.controls{background:white;border:1px solid #dce1ea;border-radius:12px;padding:16px;margin-top:14px}.buttons{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}.judge{font-size:17px;padding:13px;border:0;border-radius:9px;color:white;background:#3567d6;cursor:pointer}.judge.equal{background:#607086}.judge:hover{filter:brightness(.94)}
.row{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-bottom:13px}.row label{cursor:pointer}.skip{border:0;background:none;color:#68758c;text-decoration:underline;cursor:pointer}.hint{font-size:13px;color:#6c778b}.done{text-align:center;padding:70px;font-size:24px}.disabled{opacity:.55;pointer-events:none}@media(max-width:900px){.scores{grid-template-columns:1fr}.score{min-width:580px}}
</style></head><body><div class="top"><strong>NotaEar · 人工难度比较</strong><span id="progress">载入中…</span></div>
<main class="wrap"><div id="work"><div class="scores"><section class="card"><div class="cardhead"><h2>A</h2><audio id="left-audio" controls preload="none" aria-label="播放 A"></audio></div><div id="left" class="score"></div></section><section class="card"><div class="cardhead"><h2>B</h2><audio id="right-audio" controls preload="none" aria-label="播放 B"></audio></div><div id="right" class="score"></div></section></div>
<section class="controls"><div class="row"><strong>主要依据：</strong>
<label><input type="radio" name="dimension" value="overall" checked> 综合</label><label><input type="radio" name="dimension" value="pitch"> 音高/音程</label><label><input type="radio" name="dimension" value="rhythm"> 节奏</label><label><input type="radio" name="dimension" value="tonality"> 调性</label><label><input type="radio" name="dimension" value="notation"> 记谱/读谱</label><label><input type="radio" name="dimension" value="length"> 长度/呼吸</label></div>
<div class="row"><strong>把握：</strong><label><input type="radio" name="confidence" value="1"> 低</label><label><input type="radio" name="confidence" value="2" checked> 中</label><label><input type="radio" name="confidence" value="3"> 高</label><span class="hint">先凭实际视唱负担判断；模型预测在提交前不可见。</span></div>
<div class="buttons"><button class="judge" onclick="submitLabel('left_harder')">A 更难 <small>(1)</small></button><button class="judge equal" onclick="submitLabel('equal')">差不多 <small>(2)</small></button><button class="judge" onclick="submitLabel('right_harder')">B 更难 <small>(3)</small></button></div>
<div style="text-align:right;margin-top:10px"><button class="skip" onclick="submitLabel('skip')">看不清 / 不可比 / 跳过 (0)</button></div></section></div><div id="done" class="done" hidden></div></main>
<script>
let pair=null,busy=false,played={left:false,right:false};
function stopAudio(){for(let side of ['left','right']){let audio=document.querySelector(`#${side}-audio`);audio.pause();audio.currentTime=0}}
async function loadNext(){stopAudio();played={left:false,right:false};let r=await fetch('/api/next');let s=await r.json();if(s.done){document.querySelector('#work').hidden=true;let d=document.querySelector('#done');d.hidden=false;d.textContent=`本批 ${s.completed}/${s.total} 对已完成，可以进行下一轮训练。`;document.querySelector('#progress').textContent='完成';return}pair=s.pair;document.querySelector('#progress').textContent=`${s.completed}/${s.total} 已标 · 再标 ${s.until_retrain} 对够一轮重训`;for(let side of ['left','right']){let id=pair[side].candidate_id;let audio=document.querySelector(`#${side}-audio`);audio.src=`/audio/${id}.wav`;audio.load();let el=document.querySelector('#'+side);el.innerHTML='<div class="hint">正在排版…</div>';let svg=await (await fetch(`/score/${id}.svg`)).text();el.innerHTML=svg}}
async function submitLabel(judgement){if(!pair||busy)return;busy=true;document.querySelector('#work').classList.add('disabled');let dimension=document.querySelector('input[name=dimension]:checked').value;let confidence=Number(document.querySelector('input[name=confidence]:checked').value);let r=await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pair_id:pair.pair_id,judgement,dimension,confidence,listened:played})});if(!r.ok)alert((await r.json()).error||'保存失败');busy=false;document.querySelector('#work').classList.remove('disabled');await loadNext()}
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;if(e.key==='1')submitLabel('left_harder');if(e.key==='2')submitLabel('equal');if(e.key==='3')submitLabel('right_harder');if(e.key==='0')submitLabel('skip')});loadNext();
for(let side of ['left','right'])document.querySelector(`#${side}-audio`).addEventListener('play',()=>{played[side]=true;let other=document.querySelector(`#${side==='left'?'right':'left'}-audio`);other.pause()});
</script></body></html>'''


class LabelServer(ThreadingHTTPServer):
    def __init__(self, address, queue: dict, annotation_path: Path, annotator: str,
                 batch_size: int, audio_cache: Path, soundfont: Path):
        import verovio

        super().__init__(address, LabelHandler)
        self.queue = queue
        self.annotation_path = annotation_path
        self.annotator = annotator
        self.batch_size = batch_size
        self.lock = threading.Lock()
        self.render_lock = threading.Lock()
        self.audio_lock = threading.Lock()
        self.audio_cache = audio_cache
        self.audio_cache.mkdir(parents=True, exist_ok=True)
        self.soundfont = soundfont
        self.renderer = verovio.toolkit()
        self.renderer.setOptions({"pageWidth": 2100, "pageHeight": 1200, "scale": 34,
                                  "adjustPageHeight": True, "breaks": "auto"})
        self.pairs = {pair["pair_id"]: pair for pair in queue["pairs"]}
        self.candidates = {side["candidate_id"]: side for pair in queue["pairs"]
                           for side in (pair["left"], pair["right"])}


class LabelHandler(BaseHTTPRequestHandler):
    server: LabelServer

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def _json(self, payload, status=HTTPStatus.OK):
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            data = _html().encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/api/next":
            done = _load_done(self.server.annotation_path)
            pairs = self.server.queue["pairs"]
            pending = [(index, pair) for index, pair in enumerate(pairs) if pair["pair_id"] not in done]
            completed = len(pairs) - len(pending)
            if not pending:
                self._json({"done": True, "completed": completed, "total": len(pairs)})
            else:
                index, pair = pending[0]
                until = self.server.batch_size - (completed % self.server.batch_size)
                self._json({"done": False, "completed": completed, "total": len(pairs),
                            "until_retrain": until, "pair": _public_pair(pair, index, len(pairs))})
            return
        if path.startswith("/score/") and path.endswith(".svg"):
            candidate_id = path.removeprefix("/score/").removesuffix(".svg")
            candidate = self.server.candidates.get(candidate_id)
            if not candidate:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                with ZipFile(candidate["archive"]) as archive:
                    xml = archive.read(candidate["member"]).decode("utf-8-sig")
                with self.server.render_lock:
                    toolkit = self.server.renderer
                    if not toolkit.loadData(xml):
                        raise ValueError("Verovio could not load score")
                    data = "\n".join(toolkit.renderToSVG(page)
                                     for page in range(1, toolkit.getPageCount() + 1)).encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "private, max-age=86400")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as error:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
            return
        if path.startswith("/audio/") and path.endswith(".wav"):
            candidate_id = path.removeprefix("/audio/").removesuffix(".wav")
            candidate = self.server.candidates.get(candidate_id)
            if not candidate:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            wav_path = self.server.audio_cache / f"{candidate_id}.wav"
            try:
                with self.server.audio_lock:
                    if not wav_path.exists():
                        with ZipFile(candidate["archive"]) as archive:
                            xml = archive.read(candidate["member"]).decode("utf-8-sig")
                        with self.server.render_lock:
                            toolkit = self.server.renderer
                            if not toolkit.loadData(xml):
                                raise ValueError("Verovio could not load score")
                            midi = base64.b64decode(toolkit.renderToMIDI())
                        token = uuid.uuid4().hex
                        midi_path = self.server.audio_cache / f".{candidate_id}-{token}.mid"
                        temp_wav = self.server.audio_cache / f".{candidate_id}-{token}.wav"
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
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Cache-Control", "private, max-age=86400")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
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
            pair = self.server.pairs.get(payload.get("pair_id"))
            judgement, dimension = payload.get("judgement"), payload.get("dimension")
            confidence = int(payload.get("confidence", 0))
            listened = payload.get("listened", {})
            if not pair or judgement not in JUDGEMENTS or dimension not in DIMENSIONS or confidence not in (1, 2, 3):
                raise ValueError("invalid annotation")
            if set(listened) != {"left", "right"} or any(type(value) is not bool for value in listened.values()):
                raise ValueError("invalid listened flags")
            with self.server.lock:
                if pair["pair_id"] in _load_done(self.server.annotation_path):
                    self._json({"error": "pair already labeled"}, HTTPStatus.CONFLICT)
                    return
                def identity(side):
                    return {key: side[key] for key in ("candidate_id", "source", "member", "sha256")}
                record = {
                    "schema_version": 1,
                    "annotation_id": str(uuid.uuid4()),
                    "pair_id": pair["pair_id"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "annotator": self.server.annotator,
                    "model_version": self.server.queue["model_version"],
                    "left": identity(pair["left"]), "right": identity(pair["right"]),
                    "judgement": judgement, "dimension": dimension, "confidence": confidence,
                    "listened": listened,
                    "model_snapshot": {
                        side: {key: pair[side][key] for key in
                               ("expected_series", "uncertainty", "predicted_series", "dimension_percentiles")}
                        for side in ("left", "right")
                    },
                }
                self.server.annotation_path.parent.mkdir(parents=True, exist_ok=True)
                with self.server.annotation_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._json({"saved": True})
        except (ValueError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)


def serve(queue_path: Path, annotations: Path, host: str, port: int,
          annotator: str, batch_size: int, soundfont: Path) -> None:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    server = LabelServer((host, port), queue, annotations, annotator, batch_size,
                         queue_path.parent / "audio", soundfont)
    print(f"NotaEar labeler: http://{host}:{port}")
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
    prepare = commands.add_parser("prepare", help="select an active-learning queue")
    prepare.add_argument("--model", type=Path, default=Path("out/difficulty-melodia-v0/difficulty_model.joblib"))
    prepare.add_argument("--melodia", type=Path, default=Path("data/reference_books/raw/melodia/Cole-Lewis-1904-xml.zip"))
    prepare.add_argument("--eyes", type=Path, default=Path("data/reference_books/raw/eyes_and_ears/Crowell-2004-xml.zip"))
    prepare.add_argument("--output", type=Path, default=Path("out/active-label-v0/queue.json"))
    prepare.add_argument("--count", type=int, default=60)
    prepare.add_argument("--seed", type=int, default=20260928)
    server = commands.add_parser("serve", help="run the local labeling UI")
    server.add_argument("--queue", type=Path, default=Path("out/active-label-v0/queue.json"))
    server.add_argument("--annotations", type=Path, default=Path("data/annotations/difficulty_pairs.jsonl"))
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8765)
    server.add_argument("--annotator", default="Daniel Wu")
    server.add_argument("--batch-size", type=int, default=30)
    server.add_argument("--soundfont", type=Path,
                        default=Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"))
    args = parser.parse_args(argv)
    if args.command == "prepare":
        queue = prepare_queue(args.model, args.melodia, args.eyes, args.output, args.count, args.seed)
        print(json.dumps({"output": str(args.output), "candidates": queue["candidate_count"],
                          "pairs": len(queue["pairs"])}, ensure_ascii=False, indent=2))
    else:
        serve(args.queue, args.annotations, args.host, args.port, args.annotator,
              args.batch_size, args.soundfont)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
