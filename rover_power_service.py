"""Owns ONLY a camera/CV child process; does not power off the Pi or drive motors.

Starts in standby. Localhost HTTP control must be carried over an SSH tunnel.
An existing camera server is never adopted or killed. The supervised script
continues to own image capture/detection, preserving the teammate's CV code.
"""
import argparse
from collections import OrderedDict
from email.message import Message
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time


class MultipartFrames:
    """Bounded JPEG parts, with/without length; accepts fragmented Flask output."""
    MAX_FRAME = 4*1024*1024

    def __init__(self, content_type):
        header = Message()
        header["Content-Type"] = content_type
        boundary = header.get_param("boundary")
        if header.get_content_type() != "multipart/x-mixed-replace" or not boundary or len(boundary) > 70:
            raise ValueError("upstream must provide multipart MJPEG")
        if any(ord(c) < 32 or ord(c) >= 127 for c in boundary):
            raise ValueError("invalid MJPEG boundary")
        self.boundary = b"--" + boundary.encode("ascii")
        self.buffer = b""
        self.state = "boundary"
        self.length = None
        self.ended = False

    def feed(self, data):
        if self.ended:
            return []
        if len(self.buffer)+len(data) > self.MAX_FRAME+8192+65536:
            raise ValueError("upstream frame buffer too large")
        self.buffer += data
        frames = []
        while True:
            if self.state == "boundary":
                index = self.buffer.find(self.boundary)
                if index < 0:
                    if len(self.buffer) > 8192:
                        raise ValueError("MJPEG boundary not found")
                    break
                end = index+len(self.boundary)
                if index > 8192:
                    raise ValueError("MJPEG preamble too large")
                if len(self.buffer) < end+2:
                    break
                suffix = self.buffer[end:end+2]
                if suffix == b"--":
                    self.ended = True
                    break
                if suffix != b"\r\n":
                    raise ValueError("invalid MJPEG boundary delimiter")
                self.buffer = self.buffer[end+2:]
                self.state = "headers"
            elif self.state == "headers":
                end = self.buffer.find(b"\r\n\r\n")
                if end < 0:
                    if len(self.buffer) > 8192:
                        raise ValueError("MJPEG part headers too large")
                    break
                if end > 8192:
                    raise ValueError("MJPEG part headers too large")
                fields = {}
                for line in self.buffer[:end].split(b"\r\n"):
                    name, separator, value = line.partition(b":")
                    name = name.strip().lower()
                    if not separator or name in fields:
                        raise ValueError("invalid/duplicate MJPEG header")
                    fields[name] = value.strip()
                if fields.get(b"content-type", b"image/jpeg").lower() != b"image/jpeg":
                    raise ValueError("upstream frame is not JPEG")
                value = fields.get(b"content-length")
                self.length = None
                if value is not None:
                    if not value.isdigit() or not 4 <= int(value) <= self.MAX_FRAME:
                        raise ValueError("invalid JPEG length")
                    self.length = int(value)
                self.buffer = self.buffer[end+4:]
                self.state = "body"
            else:
                size = self.length
                if size is None:
                    size = self.buffer.find(b"\r\n"+self.boundary)
                    if size < 0:
                        if len(self.buffer) > self.MAX_FRAME+len(self.boundary)+2:
                            raise ValueError("upstream frame too large")
                        break
                elif len(self.buffer) < size:
                    break
                if not 4 <= size <= self.MAX_FRAME:
                    raise ValueError("upstream frame too large")
                frame = self.buffer[:size]
                if not frame.startswith(b"\xff\xd8") or not frame.endswith(b"\xff\xd9"):
                    raise ValueError("incomplete upstream JPEG")
                frames.append(frame)
                self.buffer = self.buffer[size:]
                self.state = "boundary"
        return frames


class Supervisor:
    def __init__(self, command, upstream_port=5000, logger=None, startup_timeout=15):
        self.command = list(command)
        self.port = upstream_port
        self.logger = logger or logging.getLogger("rover_camera")
        self.startup_timeout = startup_timeout
        self.lock = threading.RLock()
        self.process = None
        self.mode = "standby"
        self.error = ""
        self.started = 0.0
        self.seen = OrderedDict()
        self.latest_frame = None
        self.frame_sequence = 0
        self.frame_time = 0.0
        self.pumping = False

    def _port_open(self):
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=0.15):
                return True
        except OSError:
            return False

    def _log_output(self, process):
        try:
            for line in process.stdout:
                self.logger.info("camera: %s", line.rstrip()[:2048])
        finally:
            process.stdout.close()

    def _stop(self):
        process, self.process = self.process, None
        self.latest_frame = None
        self.pumping = False
        if process is not None:
            if process.poll() is None:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait(timeout=3)
                except ProcessLookupError:
                    pass
            else:
                process.wait()
        self.mode = "standby"

    def _frames(self, process):
        # Share one upstream consumer so viewers never create extra detector loops.
        connection = HTTPConnection("127.0.0.1", self.port, timeout=3)
        failure = "Camera stream ended"
        try:
            connection.request("GET", "/video_feed")
            response = connection.getresponse()
            if response.status != 200:
                raise ValueError(f"upstream HTTP {response.status}")
            parser = MultipartFrames(response.getheader("Content-Type", ""))
            while True:
                data = response.read1(65536)
                if not data:
                    break
                frames = parser.feed(data)
                if frames:
                    with self.lock:
                        if self.process is not process:
                            return
                        self.latest_frame = frames[-1]
                        self.frame_sequence += len(frames)
                        self.frame_time = time.monotonic()
                        self.mode = "active"
                if parser.ended:
                    break
        except (OSError, TimeoutError, ValueError) as exc:
            failure = f"Camera stream failed: {exc}"
        finally:
            connection.close()
            with self.lock:
                if self.process is process:
                    self._stop()
                    self.mode, self.error = "error", failure
                    self.logger.error(failure)

    def _refresh(self):
        if self.process is None:
            if self.mode == "standby" and self._port_open():
                self.mode = "error"
                self.error = "Unmanaged camera server is running; stop it before using CV standby"
            return
        code = self.process.poll()
        if code is not None:
            self.process.wait()
            self.process = None
            self.latest_frame = None
            self.pumping = False
            self.mode = "error"
            self.error = f"Camera process exited ({code}); inspect the camera-worker log"
            self.logger.error(self.error)
        elif self.mode == "starting":
            if not self.pumping and self._port_open():
                self.pumping = True
                threading.Thread(target=self._frames,args=(self.process,),daemon=True).start()
            if time.monotonic() - self.started > self.startup_timeout:
                self._stop()
                self.mode = "error"
                self.error = "Camera server did not start before its deadline"
        elif self.mode == "active" and time.monotonic()-self.frame_time > 10:
            self._stop()
            self.mode, self.error = "error", "Camera stream stopped producing frames"

    def status(self, request_id=None):
        with self.lock:
            self._refresh()
            value = {"version": 1, "type": "rover_power_status", "mode": self.mode,
                     "pi_powered": True, "controls_motors": False,
                     "camera_pid": self.process.pid if self.process else None,
                     "error": self.error}
            if request_id is not None:
                value["id"] = request_id
            return value

    def frame_after(self, sequence):
        with self.lock:
            self._refresh()
            if self.mode != "active":
                return None
            if sequence != self.frame_sequence and self.latest_frame is not None:
                return self.frame_sequence, self.latest_frame
            return ()

    def set_mode(self, mode, request_id):
        if mode not in {"standby", "active"}:
            raise ValueError("mode must be standby or active")
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
            raise ValueError("request id is required")
        with self.lock:
            self._refresh()
            if request_id in self.seen:
                if self.seen[request_id] != mode:
                    raise ValueError("request id reused with a different mode")
                return self.status(request_id)
            # Record IDs before execution to prevent duplicate or uncertain starts.
            self.seen[request_id] = mode
            while len(self.seen) > 128:
                self.seen.popitem(last=False)
            self.error = ""
            if mode == "standby":
                self._stop()
                if self._port_open():
                    self.mode = "error"
                    self.error = "Managed CV stopped, but an unmanaged camera server still owns the camera port"
            elif self.mode not in {"active", "starting"}:
                if self._port_open():
                    self.mode = "error"
                    self.error = "Camera port already in use; stop the existing server before Wake CV"
                else:
                    try:
                        kwargs = {"start_new_session": True} if os.name == "posix" else {}
                        self.process = subprocess.Popen(self.command, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", **kwargs)
                        self.started = time.monotonic()
                        self.mode = "starting"
                        threading.Thread(target=self._log_output,args=(self.process,),daemon=True).start()
                    except OSError as exc:
                        self.mode = "error"
                        self.error = f"Cannot start camera process: {exc}"
            self.logger.info("request %s mode=%s result=%s", request_id, mode, self.mode)
            return self.status(request_id)

    def close(self):
        with self.lock:
            self._stop()


def make_handler(supervisor):
    preview_lock = threading.BoundedSemaphore(1)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def _json(self, code, value):
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.connection.settimeout(5)
            if self.path == "/aerolink/status":
                self._json(200, supervisor.status())
            elif self.path == "/video_feed":
                self._preview()
            else:
                self._json(404, {"error": "unknown endpoint"})

        def do_POST(self):
            self.connection.settimeout(5)
            if self.path != "/aerolink/power":
                self._json(404, {"error": "unknown endpoint"})
                return
            # Require the GUI header and reject browser cross-origin requests.
            if self.headers.get("Origin") or self.headers.get("X-AeroLink-Control") != "1":
                self._json(403, {"error": "AeroLink control header required"})
                return
            try:
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("application/json required")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2048:
                    raise ValueError("invalid request length")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict) or type(request.get("version")) is not int or request["version"] != 1:
                    raise ValueError("invalid protocol version")
                result = supervisor.set_mode(request.get("mode"), request.get("id"))
            except (ValueError, TypeError) as exc:
                self._json(400, {"error": str(exc)})
                return
            self._json(200, result)

        def _preview(self):
            # Merely viewing the camera must never wake a sleeping CV process.
            if supervisor.status()["mode"] != "active":
                self._json(503, {"error": "CV standby or unavailable; use Wake CV"})
                return
            if not preview_lock.acquire(blocking=False):
                self._json(409, {"error": "one preview client allowed; AeroLink Larger view shares that stream"})
                return
            try:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                sequence = -1
                while True:
                    current = supervisor.frame_after(sequence)
                    if current is None:
                        break
                    if current:
                        sequence, frame = current
                        header = f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(frame)}\r\n\r\n".encode()
                        self.wfile.write(header+frame+b"\r\n")
                        self.wfile.flush()
                        time.sleep(0.1)
                    else:
                        time.sleep(0.02)
            except (OSError, TimeoutError):
                pass
            finally:
                preview_lock.release()

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--upstream-port", type=int, default=5000)
    parser.add_argument("--demo", action="store_true", help="synthetic camera process; no camera hardware")
    parser.add_argument("--script", type=Path,
        default=Path.home()/"ComputerVision/imx296_global_shutter/scripts/apriltag_stream.py")
    parser.add_argument("--python", default=sys.executable, help="existing camera virtual-environment Python")
    parser.add_argument("--log", type=Path,
        default=Path.home()/"ComputerVision/logs/aerolink-camera-worker.log")
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535 or not 1024 <= args.upstream_port <= 65535 or args.port == args.upstream_port:
        parser.error("use distinct unprivileged HTTP ports")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("rover_camera")
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(args.log,maxBytes=2*1024*1024,backupCount=2)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    if args.demo:
        script = Path(__file__).with_name("camera_preview_demo.py")
        command = [args.python,"-u",str(script),"--port",str(args.upstream_port)]
    else:
        if args.upstream_port != 5000:
            parser.error("the existing reviewed camera script listens on port 5000")
        if not args.script.is_file():
            parser.error("camera script not found; use --script or --demo")
        command = [args.python,"-u",str(args.script.resolve())]
    supervisor = Supervisor(command,args.upstream_port,logger)
    try:
        server = ThreadingHTTPServer(("127.0.0.1",args.port),make_handler(supervisor))
    except OSError:
        handler.close()
        raise
    server.daemon_threads = True
    print(f"AeroLink CV service on 127.0.0.1:{args.port}; starts in STANDBY",flush=True)
    print("Software standby stops only its own camera/CV child; Pi/network stay powered.",flush=True)
    if args.demo:
        print("SIMULATED camera mode",flush=True)
    def stop_signal(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop_signal)
    stopped = threading.Event()
    def monitor():
        while not stopped.wait(0.25):
            supervisor.status()
    monitor_thread = threading.Thread(target=monitor,daemon=True)
    monitor_thread.start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        stopped.set()
        server.server_close()
        supervisor.close()
        handler.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
