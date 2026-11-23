"""Synthetic vision samples only: exercises protocol, not a real vision model.

Use locally with UDP, or execute via the GUI SSH remote command with --stdout.
"""
import argparse
import json
import socket
import time


def sample(sequence, elapsed):
    return {
        "type": "vision", "sequence": sequence, "timestamp_ms": int(time.time()*1000),
        "target_detected": elapsed >= 2, "tag_visible": elapsed >= 4,
        "confidence": 0.95 if elapsed >= 2 else 0,
        "x_m": 0.3 if elapsed < 6 else 0.03, "y_m": 0.02,
        "yaw_deg": 12 if elapsed < 6 else 2,
        "distance_m": 1.0 if elapsed < 10 else 0.2,
        "capture_confirmed": elapsed >= 14, "latch_connected": elapsed >= 14, "retry_count": 0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port",type=int,default=14600)
    p.add_argument("--stdout",action="store_true")
    p.add_argument("--seconds",type=float,default=60)
    args = p.parse_args()
    if not 1024 <= args.port <= 65535 or args.seconds <= 0:
        p.error("invalid port or duration")
    start = time.monotonic()
    sequence = 0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            while time.monotonic()-start < args.seconds:
                message = json.dumps(sample(sequence,time.monotonic()-start))
                if args.stdout:
                    print(message,flush=True)
                else:
                    sock.sendto(message.encode(),("127.0.0.1",args.port))
                sequence += 1
                time.sleep(0.25)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
