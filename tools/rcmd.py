#!/usr/bin/env python3
"""Run a shell command on the edge device over SSH (password auth).

Usage:
  export EDGE_HOST=192.0.2.10 EDGE_USER=bianbu EDGE_PASS=...
  python rcmd.py "command"                 # stdout/stderr streamed to console
  python rcmd.py -t 3600 "long command"    # custom timeout (seconds)
  python rcmd.py -f script.sh              # upload+run a local script
  python rcmd.py -u src dst                # upload a file (scp-like)
  python rcmd.py -d src dst                # download a file

The host and password come from the environment only; nothing is baked in here.
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import paramiko

HOST = os.environ.get("EDGE_HOST")
PORT = int(os.environ.get("EDGE_PORT", "22"))
USER = os.environ.get("EDGE_USER", "bianbu")
PASS = os.environ.get("EDGE_PASS")

if not HOST or not PASS:
    sys.exit("set EDGE_HOST and EDGE_PASS (and optionally EDGE_USER/EDGE_PORT)")


def connect(timeout=30):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(
        HOST,
        port=PORT,
        username=USER,
        password=PASS,
        timeout=timeout,
        banner_timeout=60,
        auth_timeout=60,
        look_for_keys=False,
        allow_agent=False,
    )
    return c


def run(cmd, timeout=600, quiet=False):
    c = connect()
    try:
        stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout, get_pty=False)
        out = []
        for line in stdout:
            out.append(line)
            if not quiet:
                sys.stdout.write(line)
                sys.stdout.flush()
        err = stdout.channel.recv_stderr(65536) if False else b""
        for line in stderr:
            err += line if isinstance(line, bytes) else line.encode()
            if not quiet:
                sys.stderr.write(line if isinstance(line, str) else line.decode("utf-8", "replace"))
                sys.stderr.flush()
        rc = stdout.channel.recv_exit_status()
        return rc, "".join(out), err.decode("utf-8", "replace")
    finally:
        c.close()


def sftp_put(local, remote, quiet=False):
    """Push a file over SFTP with progress. Windows firewall blocks the device
    from reaching back to us, so we always push rather than let it pull."""
    import time

    c = connect()
    try:
        t = c.get_transport()
        t.set_keepalive(30)
        sftp = paramiko.SFTPClient.from_transport(t)
        sftp.get_channel().settimeout(3600)
        size = os.path.getsize(local)
        state = {"last": 0.0}

        def cb(done, total):
            now = time.time()
            if now - state["last"] >= 2 or done == total:
                state["last"] = now
                pct = 100.0 * done / total
                sys.stderr.write("\r  %6.2f%%  %7.1f / %7.1f MB" % (pct, done / 1e6, total / 1e6))
                sys.stderr.flush()

        t0 = time.time()
        sftp.put(local, remote, callback=cb, confirm=True)
        dt = time.time() - t0
        if not quiet:
            sys.stderr.write("\n  %.1fs  %.1f MB/s\n" % (dt, size / 1e6 / dt))
        sftp.close()
    finally:
        c.close()


def put(local, remote, timeout=3600):
    """Push a local file to the device.

    Small files go through the exec channel as base64; anything sizeable uses
    SFTP, which is several times faster.
    """
    import base64

    if os.path.getsize(local) > 4 * 1024 * 1024:
        run_quiet("mkdir -p %s" % (os.path.dirname(remote) or "."))
        return sftp_put(local, remote)

    with open(local, "rb") as fh:
        payload = base64.b64encode(fh.read()).decode("ascii")

    c = connect()
    try:
        cmd = "mkdir -p %s && base64 -d > %s" % (os.path.dirname(remote) or ".", remote)
        stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
        for i in range(0, len(payload), 65536):
            stdin.write(payload[i : i + 65536])
        stdin.channel.shutdown_write()
        rc = stdout.channel.recv_exit_status()
        err = stderr.read().decode("utf-8", "replace")
        if rc != 0:
            raise RuntimeError("upload failed rc=%d: %s" % (rc, err))
    finally:
        c.close()


def run_quiet(cmd, timeout=120):
    c = connect()
    try:
        _, stdout, _ = c.exec_command(cmd, timeout=timeout)
        stdout.channel.recv_exit_status()
    finally:
        c.close()


def get(remote, local, timeout=3600):
    c = connect()
    try:
        sftp = c.open_sftp()
        sftp.get(remote, local)
        sftp.close()
    finally:
        c.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--timeout", type=int, default=600)
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("-f", "--file", help="upload local script and run it")
    ap.add_argument("-u", "--upload", nargs=2, metavar=("LOCAL", "REMOTE"))
    ap.add_argument("-d", "--download", nargs=2, metavar=("REMOTE", "LOCAL"))
    ap.add_argument("cmd", nargs="*")
    a = ap.parse_args()

    if a.upload:
        put(a.upload[0], a.upload[1])
        print("uploaded %s -> %s" % tuple(a.upload))
        return
    if a.download:
        get(a.download[0], a.download[1])
        print("downloaded %s -> %s" % tuple(a.download))
        return
    if a.file:
        base = os.path.basename(a.file)
        put(a.file, "/tmp/" + base)
        cmd = "bash /tmp/%s" % base
    else:
        cmd = " ".join(a.cmd)
    if not cmd:
        ap.error("no command")
    rc, _, _ = run(cmd, timeout=a.timeout, quiet=a.quiet)
    sys.exit(rc)


if __name__ == "__main__":
    main()
