"""The whole presenter, live in a browser tab, seen through the physical cameras.

This is the application - the real `BehaviorEngine` on a real clock - rendered
from the canonical 3D world and streamed as MJPEG. The seven cameras in
`config/cameras.yaml` are the viewpoints, so switching camera is a switch of
projection, not a switch of picture.

    python tools/web_app.py            # then open http://127.0.0.1:8770

## Why this can be live at all

The world is built once (~0.9 s) and then only re-posed: the head rotates about
the neck pivot and the chest breathes, which is a handful of transforms rather
than a rebuild. Rebuilding the human every frame costs about as much as
rendering it and would halve the frame rate for nothing, since the room, desk,
chair, lights and cameras have not moved.

EEVEE then renders 640x360 at roughly 10 fps. That is not 30, and the readout
says so rather than implying otherwise.

## What is live and what is not

Live: head yaw/pitch/roll, blinks, breathing, behaviour state, camera choice.
Not live: gaze direction, which the 3D proxy aims at a fixed target rather than
at `pose.gaze_x/y`, and identity, which is the debug mannequin until the
reconstruction exists. Both are stated in the overlay instead of being left for
someone to discover.

Editing config/cameras.yaml or the scene builder rebuilds the world in place -
no restart.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

WATCH = [
    ROOT / "config" / "cameras.yaml",
    ROOT / "config" / "room_geometry.yaml",
    ROOT / "src" / "presenter" / "scene3d" / "world.py",
]

shared = {
    "jpeg": None,
    "seq": 0,
    "camera": "cam1",
    "state": "IDLE_ATTENTIVE",
    "fps": 0.0,
    "render_ms": 0.0,
    "elapsed": 0.0,
    "pose": {},
    "cameras": [],
    "note": "",
    "rebuilding": False,
}
lock = threading.Condition()
want = {"camera": "cam1", "state": None, "rebuild": False}


def fingerprint():
    out = []
    for p in WATCH:
        try:
            st = p.stat()
            out.append((p.name, int(st.st_mtime_ns), st.st_size))
        except FileNotFoundError:
            out.append((p.name, 0, 0))
    return out


def engine_loop(width, height, samples, fps_cap):
    """Behaviour -> pose -> re-pose the world -> render -> publish a JPEG."""
    import bpy

    from presenter.behavior import BehaviorEngine
    from presenter.render.camera_manager import CameraManager
    from presenter.behavior.state import BehaviorState
    from presenter.scene3d.world import build_world

    engine = BehaviorEngine(seed=7)
    pose = engine.update(1.0 / 30.0)

    def build():
        w = build_world(pose)
        scn = bpy.context.scene
        scn.render.engine = "BLENDER_EEVEE"
        scn.render.resolution_x, scn.render.resolution_y = width, height
        scn.render.resolution_percentage = 100
        scn.render.image_settings.file_format = "JPEG"
        scn.render.image_settings.quality = 80
        # Identical to the stills, so a live frame and a rendered preview of the
        # same camera are the same picture.
        scn.view_settings.view_transform = "Filmic"
        scn.view_settings.look = "None"
        scn.view_settings.exposure = 0.0
        if hasattr(scn, "eevee"):
            scn.eevee.taa_render_samples = samples
        return w, scn

    world, scn = build()
    mgr = CameraManager.load("config/cameras.yaml")
    with lock:
        shared["cameras"] = [
            {"key": v.key, "intent": v.intent, "focal": v.focal_mm,
             "position": [round(c, 2) for c in v.position],
             "enabled": v.enabled}
            for v in mgr.ordered()
        ]

    tmp = ROOT / "renders" / "_live.jpg"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    scn.render.filepath = str(tmp.with_suffix(""))

    last_fp = fingerprint()
    frame_times = []
    last = time.perf_counter()
    states = {s.value: s for s in BehaviorState}

    while True:
        now = time.perf_counter()
        dt = min(now - last, 0.25)
        last = now

        # Config or scene edits rebuild in place rather than needing a restart.
        fp = fingerprint()
        if fp != last_fp or want["rebuild"]:
            last_fp = fp
            want["rebuild"] = False
            with lock:
                shared["rebuilding"] = True
                shared["note"] = "rebuilding world..."
                lock.notify_all()
            try:
                world, scn = build()
                mgr = CameraManager.load("config/cameras.yaml")
                with lock:
                    shared["cameras"] = [
                        {"key": v.key, "intent": v.intent, "focal": v.focal_mm,
                         "position": [round(c, 2) for c in v.position],
                         "enabled": v.enabled}
                        for v in mgr.ordered()
                    ]
                    shared["note"] = "world rebuilt from config"
                scn.render.filepath = str(tmp.with_suffix(""))
            except Exception as exc:  # noqa: BLE001
                with lock:
                    shared["note"] = f"rebuild failed: {type(exc).__name__}: {exc}"
            finally:
                with lock:
                    shared["rebuilding"] = False

        if want["state"] is not None:
            s = states.get(want["state"])
            want["state"] = None
            if s is not None:
                engine.set_state(s)

        pose = engine.update(dt)
        world.repose(pose)

        cam_ob = world.cameras.get(want["camera"])
        if cam_ob is not None:
            scn.camera = cam_ob

        t0 = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        render_ms = (time.perf_counter() - t0) * 1000.0

        try:
            data = tmp.read_bytes()
        except OSError:
            continue

        frame_times.append(time.perf_counter())
        frame_times[:] = [t for t in frame_times if t > time.perf_counter() - 3.0]

        with lock:
            shared["jpeg"] = data
            shared["seq"] += 1
            shared["camera"] = want["camera"]
            shared["state"] = pose.state if isinstance(pose.state, str) else str(pose.state)
            shared["fps"] = len(frame_times) / 3.0
            shared["render_ms"] = render_ms
            shared["elapsed"] = engine.stats.elapsed
            shared["pose"] = {
                "yaw": round(pose.yaw, 2), "pitch": round(pose.pitch, 2),
                "roll": round(pose.roll, 2),
                "eye_l": round(pose.eye_open_l, 3),
                "eye_r": round(pose.eye_open_r, 3),
                "breath": round(pose.scale, 4),
                "blinks": engine.stats.blinks,
            }
            lock.notify_all()

        if fps_cap:
            slack = (1.0 / fps_cap) - (time.perf_counter() - now)
            if slack > 0:
                time.sleep(slack)


PAGE = """<!doctype html><meta charset=utf-8>
<title>Presenter - live</title>
<style>
 :root{--bg:#0f1116;--panel:#171a21;--line:#282d38;--tx:#e2e7f0;--dim:#889;--ok:#5ad98d;--warn:#fc6}
 *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);
   font:14px/1.5 ui-sans-serif,system-ui,Segoe UI,sans-serif}
 header,footer{background:var(--panel);border-color:var(--line);padding:9px 13px;
   display:flex;gap:7px;align-items:center;flex-wrap:wrap}
 header{border-bottom:1px solid var(--line);position:sticky;top:0;z-index:9}
 footer{border-top:1px solid var(--line);font:12px ui-monospace,Consolas,monospace;color:var(--dim)}
 button{background:#212734;color:var(--tx);border:1px solid var(--line);border-radius:7px;
   padding:6px 10px;font:inherit;cursor:pointer}
 button:hover{border-color:#3d94ff}
 button.on{background:#123f6d;border-color:#3d94ff}
 .sp{flex:1}.tag{color:var(--dim);font-size:12px;margin-right:4px}
 #view{display:block;margin:0 auto;max-width:100%;background:#000}
 .live{color:var(--ok)}.warn{color:var(--warn)}
</style>
<header id=cams></header>
<img id=view src="/stream">
<header id=states style="position:static;border-top:1px solid var(--line)"></header>
<footer id=hud></footer>
<script>
let cams=[],active='cam1';
const STATES=['IDLE_ATTENTIVE','IDLE_RELAXED','LISTENING','THINKING','SPEAKING','READING','FOCUSED','MILD_POSITIVE','MILD_CONCERN'];

function drawCams(){
  const h=document.getElementById('cams');h.innerHTML='';
  const t=document.createElement('span');t.className='tag';t.textContent='CAMERA';h.appendChild(t);
  cams.forEach((c,i)=>{const b=document.createElement('button');
    b.textContent=(i+1)+'  '+c.key.toUpperCase()+'  '+c.intent;
    b.className=c.key===active?'on':'';
    b.onclick=()=>pick(c.key);h.appendChild(b);});
}
function drawStates(){
  const h=document.getElementById('states');h.innerHTML='';
  const t=document.createElement('span');t.className='tag';t.textContent='BEHAVIOUR';h.appendChild(t);
  STATES.forEach(s=>{const b=document.createElement('button');
    b.textContent=s.replace('_',' ');b.onclick=()=>fetch('/api/state/'+s,{method:'POST'});
    h.appendChild(b);});
}
async function pick(k){active=k;drawCams();await fetch('/api/camera/'+k,{method:'POST'});}
addEventListener('keydown',e=>{const n=parseInt(e.key,10);
  if(n>=1&&n<=cams.length)pick(cams[n-1].key);
  else if(e.key==='[')pick(cams[(cams.findIndex(c=>c.key===active)-1+cams.length)%cams.length].key);
  else if(e.key===']')pick(cams[(cams.findIndex(c=>c.key===active)+1)%cams.length].key);});

const es=new EventSource('/api/events');
es.onmessage=ev=>{
  const s=JSON.parse(ev.data);
  if(JSON.stringify(s.cameras)!==JSON.stringify(cams)){cams=s.cameras;drawCams();}
  if(s.camera!==active){active=s.camera;drawCams();}
  const p=s.pose||{};
  document.getElementById('hud').innerHTML=
    '<span class="'+(s.rebuilding?'warn':'live')+'">&#9679;</span>&nbsp;'+
    (s.rebuilding?'REBUILDING':'LIVE')+
    '&nbsp; '+s.fps.toFixed(1)+' fps &nbsp; render '+s.render_ms.toFixed(0)+' ms'+
    ' &nbsp;|&nbsp; '+s.camera.toUpperCase()+
    ' &nbsp;|&nbsp; state '+s.state+
    ' &nbsp;|&nbsp; yaw '+p.yaw+' pitch '+p.pitch+' roll '+p.roll+
    ' &nbsp; lids '+p.eye_l+'/'+p.eye_r+' &nbsp; breath '+p.breath+
    ' &nbsp; blinks '+p.blinks+' &nbsp; t '+s.elapsed.toFixed(1)+'s'+
    ' &nbsp;|&nbsp; PROXY HUMAN, gaze not driven in 3D'+
    (s.note?' &nbsp;|&nbsp; '+s.note:'');
};
drawStates();
</script>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # noqa: A003
        pass

    def _send(self, code, body, ctype):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        if self.path.startswith("/api/camera/"):
            want["camera"] = self.path.rsplit("/", 1)[-1]
        elif self.path.startswith("/api/state/"):
            want["state"] = self.path.rsplit("/", 1)[-1]
        elif self.path == "/api/rebuild":
            want["rebuild"] = True
        self._send(200, "{}", "application/json")

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]

        if path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
            return

        if path == "/stream":
            # MJPEG. One TCP connection, frames pushed as they are rendered -
            # no polling, and the browser shows them as fast as they arrive.
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=f")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seen = -1
            try:
                while True:
                    with lock:
                        while shared["seq"] == seen or shared["jpeg"] is None:
                            lock.wait(timeout=5.0)
                        seen = shared["seq"]
                        buf = shared["jpeg"]
                    out = io.BytesIO()
                    out.write(b"--f\r\nContent-Type: image/jpeg\r\n")
                    out.write(f"Content-Length: {len(buf)}\r\n\r\n".encode())
                    out.write(buf)
                    out.write(b"\r\n")
                    self.wfile.write(out.getvalue())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        if path == "/api/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    with lock:
                        lock.wait(timeout=2.0)
                        payload = json.dumps({k: v for k, v in shared.items()
                                              if k != "jpeg"})
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        self._send(404, b"no", "text/plain")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--fps", type=float, default=0.0,
                    help="cap the loop; 0 renders as fast as EEVEE allows")
    args = ap.parse_args()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"\n  presenter, live   http://{args.host}:{args.port}")
    print(f"  render            {args.width}x{args.height}, {args.samples} samples")
    print("  cameras           config/cameras.yaml, all seven, keys 1-7")
    print("  edits             cameras.yaml / room_geometry.yaml / world.py "
          "rebuild in place\n")

    # The HTTP server is what goes in a thread, not the renderer.
    #
    # bpy is not thread-safe, and `bpy.ops.render.render()` called off the main
    # thread does not raise - it returns a flat grey frame, which looks exactly
    # like a camera aimed at a wall and sent me looking at the camera transforms
    # instead of at the threading. The render loop owns the main thread.
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        engine_loop(args.width, args.height, args.samples, args.fps)
    except KeyboardInterrupt:
        print("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
