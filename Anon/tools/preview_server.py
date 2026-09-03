"""Live camera preview in the browser. Edit a config, watch the shot change.

The camera work is file-driven - `config/cameras.yaml` in, seven PNGs out - so
the edit/restart cycle around an OpenCV window is pure overhead. This serves the
same seven stills over HTTP, watches the files they are derived from, and
re-renders and pushes an update when any of them changes. Nothing to restart.

    python tools/preview_server.py            # then open http://127.0.0.1:8765
    python tools/preview_server.py --no-render    # watch only, never call Blender

Stdlib only, on purpose. Flask and watchdog are not installed, and the venv is
shared with the movement terminal - adding packages to it to get a dev server is
not a trade worth making.

## What triggers what

    config/cameras.yaml, config/room_geometry.yaml
        -> re-render all seven previews (Blender, ~8 s), then reload the page
    renders/camera_preview/*.png
        -> reload the page, no render

The render runs in a worker thread so the page stays responsive and can say
"rendering" rather than appearing to hang.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PREVIEW = ROOT / "renders" / "camera_preview"
# Everything a preview is derived from. The two configs are the obvious ones,
# but the scene builder and the render script change the picture just as
# directly, and having to remember which kind of edit needs a manual re-render
# is exactly the friction this server exists to remove.
SOURCES = [
    ROOT / "config" / "cameras.yaml",
    ROOT / "config" / "room_geometry.yaml",
    ROOT / "src" / "presenter" / "scene3d" / "world.py",
    ROOT / "tools" / "render_camera_previews.py",
    ROOT / "tools" / "camera_layout.py",
]

state = {
    "version": 0,
    "status": "ready",
    "message": "",
    "active": "cam1",
    "at": time.strftime("%H:%M:%S"),
}
_lock = threading.Condition()


def bump(status="ready", message=""):
    with _lock:
        state["version"] += 1
        state["status"] = status
        state["message"] = message
        state["at"] = time.strftime("%H:%M:%S")
        _lock.notify_all()


def cameras():
    from presenter.render.camera_manager import CameraManager

    try:
        mgr = CameraManager.load("config/cameras.yaml")
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)
    out = []
    for v in mgr.ordered():
        p, _, y = v.rotation_deg
        out.append({
            "key": v.key, "intent": v.intent, "focal": v.focal_mm,
            "position": [round(c, 2) for c in v.position],
            "look_at": [round(c, 2) for c in v.look_at],
            "pitch": p, "yaw": y, "enabled": v.enabled,
            "has_preview": v.has_preview(),
        })
    return out, ""


def fingerprint():
    """mtime+size of everything a preview depends on."""
    parts = []
    for p in SOURCES + sorted(PREVIEW.glob("cam*.png")) + [PREVIEW / "contact_sheet.png",
                                                           PREVIEW / "camera_floorplan_debug.png"]:
        try:
            st = p.stat()
            parts.append((str(p), int(st.st_mtime_ns), st.st_size))
        except FileNotFoundError:
            parts.append((str(p), 0, 0))
    return parts


def config_fingerprint():
    return [f for f in fingerprint() if any(str(s) == f[0] for s in SOURCES)]


def render(script, label):
    bump("working", f"running {label}...")
    try:
        r = subprocess.run([sys.executable, str(ROOT / "tools" / script), "--force"]
                           if script == "render_camera_previews.py"
                           else [sys.executable, str(ROOT / "tools" / script)],
                           capture_output=True, text=True, cwd=str(ROOT), timeout=900)
        if r.returncode != 0:
            tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
            bump("error", f"{label} failed: {' | '.join(tail)}")
            return False
    except Exception as exc:  # noqa: BLE001
        bump("error", f"{label}: {type(exc).__name__} {exc}")
        return False
    return True


def watcher(auto_render: bool):
    """Poll, because watchdog is not installed and 0.5 s is fast enough here."""
    last, last_cfg = fingerprint(), config_fingerprint()
    while True:
        time.sleep(0.5)
        now_cfg = config_fingerprint()
        if now_cfg != last_cfg:
            # Name the changed files BEFORE last_cfg is advanced, or the
            # comparison below is against itself and always reports nothing.
            changed = [Path(f[0]).name for f, g in zip(now_cfg, last_cfg)
                       if f != g] or ["source"]
            last_cfg = now_cfg
            if auto_render:
                # A camera moved. Re-render the stills, then the floorplan, so
                # the plan view and the shots never disagree on screen.
                if render("render_camera_previews.py", "preview render"):
                    render("camera_layout.py", "floorplan")
                    bump("ready", f"{', '.join(changed)} changed - re-rendered")
            else:
                bump("ready", "config changed (auto-render off)")
            last = fingerprint()
            continue

        now = fingerprint()
        if now != last:
            last = now
            bump("ready", "preview images changed on disk")


PAGE = """<!doctype html><meta charset=utf-8>
<title>Camera preview</title>
<style>
 :root{--bg:#111318;--panel:#181b22;--line:#2a2f3a;--tx:#dfe4ee;--dim:#8b93a4;--ok:#59d98b;--warn:#ffcc66;--err:#ff7a7a}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--tx);
   font:14px/1.5 ui-sans-serif,system-ui,Segoe UI,sans-serif}
 header{display:flex;gap:8px;align-items:center;flex-wrap:wrap;
   padding:10px 14px;background:var(--panel);border-bottom:1px solid var(--line);
   position:sticky;top:0;z-index:5}
 button{background:#222734;color:var(--tx);border:1px solid var(--line);
   border-radius:7px;padding:7px 11px;font:inherit;cursor:pointer}
 button:hover{border-color:#3d94ff} button.on{background:#12406e;border-color:#3d94ff}
 button.no{opacity:.45}
 #shot{display:block;max-width:100%;margin:0 auto;background:#000}
 #meta{padding:8px 14px;color:var(--dim);font:12px ui-monospace,Consolas,monospace;
   border-top:1px solid var(--line);background:var(--panel)}
 .dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:6px}
 .ready{background:var(--ok)} .working{background:var(--warn)} .error{background:var(--err)}
 .sp{flex:1}
</style>
<header id=bar></header>
<img id=shot alt="camera preview">
<div id=meta></div>
<script>
let cams=[], active="cam1", view="camera", ver=0;

async function load(){
  const s = await (await fetch('/state')).json();
  cams = s.cameras; active = s.active; ver = s.version;
  draw(s);
}
function draw(s){
  const bar=document.getElementById('bar'); bar.innerHTML='';
  cams.forEach((c,i)=>{
    const b=document.createElement('button');
    b.textContent=(i+1)+'  '+c.key.toUpperCase()+'  '+c.intent;
    b.className=(view==='camera'&&c.key===active?'on':'')+(c.has_preview?'':' no');
    b.onclick=()=>pick(c.key);
    bar.appendChild(b);
  });
  const sp=document.createElement('span'); sp.className='sp'; bar.appendChild(sp);
  [['floorplan','F  FLOORPLAN'],['sheet','C  CONTACT SHEET']].forEach(([v,label])=>{
    const b=document.createElement('button');
    b.textContent=label; b.className=view===v?'on':''; b.onclick=()=>{view=v;render(s)};
    bar.appendChild(b);
  });
  render(s);
}
function render(s){
  const img=document.getElementById('shot');
  const bust='?v='+ver+'&t='+Date.now();
  img.src = view==='camera' ? '/img/'+active+'.png'+bust
          : view==='floorplan' ? '/img/camera_floorplan_debug.png'+bust
          : '/img/contact_sheet.png'+bust;
  const c=cams.find(x=>x.key===active)||{};
  const m=document.getElementById('meta');
  const st=s.status||'ready';
  m.innerHTML='<span class="dot '+st+'"></span>'+
    (view==='camera'
      ? c.key+'  '+c.intent+'   focal '+c.focal+'mm   pos ['+ (c.position||[]).join(', ')+
        ']   aim ['+(c.look_at||[]).join(', ')+']   pitch '+c.pitch+'  yaw '+c.yaw+
        (c.enabled?'':'   [production-blocked]')+(c.has_preview?'':'   NO PREVIEW RENDERED')
      : view)+
    '   —   v'+ver+'  '+(s.at||'')+'  '+(s.message||'');
}
async function pick(k){
  active=k; view='camera';
  await fetch('/select/'+k,{method:'POST'});
  const s=await (await fetch('/state')).json(); draw(s);
}
addEventListener('keydown',e=>{
  const n=parseInt(e.key,10);
  if(n>=1&&n<=cams.length) pick(cams[n-1].key);
  else if(e.key==='f') {view='floorplan';fetch('/state').then(r=>r.json()).then(render);}
  else if(e.key==='c') {view='sheet';fetch('/state').then(r=>r.json()).then(render);}
  else if(e.key==='[') pick(cams[(cams.findIndex(c=>c.key===active)-1+cams.length)%cams.length].key);
  else if(e.key===']') pick(cams[(cams.findIndex(c=>c.key===active)+1)%cams.length].key);
});
const es=new EventSource('/events');
es.onmessage=async ev=>{
  const s=JSON.parse(ev.data); ver=s.version;
  const fresh=await (await fetch('/state')).json();
  cams=fresh.cameras; draw(fresh);
};
load();
</script>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # noqa: A003 - the console is for the watcher
        pass

    def _send(self, code, body, ctype, extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        if self.path.startswith("/select/"):
            key = self.path.rsplit("/", 1)[-1]
            with _lock:
                state["active"] = key
            self._send(200, "{}", "application/json")
            return
        self._send(404, "no", "text/plain")

    def do_GET(self):  # noqa: N802, C901
        path = self.path.split("?", 1)[0]

        if path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
            return

        if path == "/state":
            cams, err = cameras()
            with _lock:
                payload = dict(state)
            payload["cameras"] = cams
            payload["error"] = err
            self._send(200, json.dumps(payload), "application/json")
            return

        if path.startswith("/img/"):
            name = Path(path).name
            f = PREVIEW / name
            if not f.exists():
                self._send(404, b"missing", "text/plain")
                return
            ctype = mimetypes.guess_type(str(f))[0] or "image/png"
            self._send(200, f.read_bytes(), ctype)
            return

        if path == "/events":
            # Server-sent events: one line per change, held open. This is why
            # the server is threading - a blocked stream must not block the
            # image requests behind it.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            seen = -1
            try:
                while True:
                    with _lock:
                        if state["version"] == seen:
                            _lock.wait(timeout=15.0)
                        seen = state["version"]
                        payload = json.dumps(dict(state))
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
            return

        self._send(404, b"no", "text/plain")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-render", action="store_true",
                    help="watch files but never invoke Blender")
    args = ap.parse_args()

    if not PREVIEW.exists() or not list(PREVIEW.glob("cam*.png")):
        print("No previews yet - rendering once before serving.")
        render("render_camera_previews.py", "preview render")

    threading.Thread(target=watcher, args=(not args.no_render,),
                     daemon=True).start()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"\n  camera preview   http://{args.host}:{args.port}")
    print(f"  watching         config/cameras.yaml, config/room_geometry.yaml")
    print(f"  auto-render      {'off' if args.no_render else 'on'}")
    print("\n  Edit a camera position and the page updates itself. Ctrl+C to stop.\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
