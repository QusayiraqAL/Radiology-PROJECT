# -*- coding: utf-8 -*-
"""
Live progress panel for whatever training run is currently going.

Answers one question at a glance: how much longer? The training log prints `ep 11/14` but no
timestamps, so elapsed time comes from the OS — the start time of the running python.exe —
and the rate is derived from it: seconds_per_epoch = elapsed / epochs_done.

Two honest caveats are printed rather than hidden:
  * ETA assumes every remaining epoch runs. Early stopping (patience) can cut a run short, so
    the estimate is an UPPER bound, labelled `<=`.
  * Stage A (frozen backbone) epochs are much cheaper than stage B (full fine-tune). While a
    run is still in stage A the average is optimistic; once it is in stage B the ETA is
    computed from stage-B epochs only, which is the number that matters.

  python progress.py              # one snapshot
  python progress.py -w           # refresh every 20s
  python progress.py -w 5         # refresh every 5s
"""
import os, re, sys, json, time, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "models", "_retrain91.log")

RE_HEAD = re.compile(r"^\[([A-Za-z0-9_]+)\] ([a-z]+mnist) size=(\d+) classes=(\d+)")
RE_EP = re.compile(r"^\s+ep\s+(\d+)/(\d+) \[([^\]]+)\].*?val_acc=([\d.]+)")
RE_RES = re.compile(r"^\[RESULT\] ([A-Za-z0-9_]+) test_acc=([\d.]+)")
RE_STOP = re.compile(r"^\s+\[early stop\]")
RE_FATAL = re.compile(r"^\[FATAL\]")
RE_MEMBER = re.compile(r"^\[ens\] ---- training member seed=(\d+) -> ([A-Za-z0-9_]+) ----")

C = {"g": "\033[92m", "y": "\033[93m", "r": "\033[91m", "b": "\033[94m",
     "d": "\033[90m", "B": "\033[1m", "0": "\033[0m"}
if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
    C = {k: "" for k in C}


def hms(sec):
    if sec is None or sec < 0 or sec != sec:
        return "  --  "
    sec = int(sec)
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return ("%dh%02dm" % (h, m)) if h else ("%dm%02ds" % (m, s)) if m else ("%ds" % s)


def ps_python():
    """Running python processes with start time and command line, newest first."""
    cmd = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
           "Select-Object ProcessId,CreationDate,WorkingSetSize,CommandLine | ConvertTo-Json")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                             capture_output=True, text=True, timeout=25).stdout
        data = json.loads(out) if out.strip() else []
    except Exception:
        return []
    if isinstance(data, dict):
        data = [data]
    rows = []
    for d in data:
        cd = d.get("CreationDate")
        ts = None
        if isinstance(cd, str):
            m = re.search(r"/Date\((\d+)", cd)          # \/Date(1757...)\/
            if m:
                ts = int(m.group(1)) / 1000.0
            else:
                try:
                    ts = time.mktime(time.strptime(cd[:19], "%Y-%m-%dT%H:%M:%S"))
                except Exception:
                    ts = None
        rows.append({"pid": d.get("ProcessId"), "start": ts,
                     "mb": (d.get("WorkingSetSize") or 0) / 1048576.0,
                     "cmd": d.get("CommandLine") or ""})
    return sorted(rows, key=lambda r: -(r["mb"] or 0))


def free_mb():
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "$o=Get-CimInstance Win32_OperatingSystem;"
             "'{0} {1}' -f $o.FreePhysicalMemory,$o.TotalVisibleMemorySize"],
            capture_output=True, text=True, timeout=20).stdout.split()
        return int(out[0]) / 1024.0, int(out[1]) / 1024.0
    except Exception:
        return None, None


def parse():
    """Walk the log once and return (finished_jobs, active_job_or_None)."""
    if not os.path.exists(LOG):
        return [], None
    with open(LOG, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    done, cur = [], None
    for ln in lines:
        m = RE_MEMBER.match(ln)
        if m:
            continue                                  # header for the member follows anyway
        m = RE_HEAD.match(ln)
        if m:
            cur = {"key": m.group(1), "dataset": m.group(2), "size": int(m.group(3)),
                   "classes": int(m.group(4)), "ep": 0, "total": 0, "stage": "-",
                   "val": None, "best": None, "stageB_first": None, "eps": []}
            continue
        if cur is None:
            continue
        m = RE_EP.match(ln)
        if m:
            cur["ep"], cur["total"] = int(m.group(1)), int(m.group(2))
            cur["stage"] = m.group(3)
            cur["val"] = float(m.group(4))
            cur["best"] = max(cur["best"] or 0, cur["val"])
            cur["eps"].append((cur["ep"], cur["stage"]))
            if cur["stage"].startswith("B") and cur["stageB_first"] is None:
                cur["stageB_first"] = cur["ep"]
            continue
        m = RE_RES.match(ln)
        if m:
            # Trust the key printed on the RESULT line, not whichever header happened to be
            # last. Session 2 ran two runners at once and their output is interleaved line by
            # line, so a sequential "last header wins" attribution silently mislabels results
            # in that region (it credited retina_bin_s3's 0.8475 to s4).
            cur["result"] = float(m.group(2))
            cur["state"] = "done"
            if m.group(1) != cur["key"]:
                cur["key"] = m.group(1)
                cur["interleaved"] = True
            done.append(cur); cur = None
            continue
        if RE_FATAL.match(ln) and cur is not None:
            cur["state"] = "fatal"
            done.append(cur); cur = None
            continue
        if RE_STOP.match(ln) and cur is not None:
            cur["early_stopped"] = True
    return done, cur


ANSI = re.compile(r"\033\[[0-9;]*m")


def row(s, w):
    """Pad or truncate to exactly `w` VISIBLE columns, ignoring ANSI colour codes."""
    vis = ANSI.sub("", s)
    if len(vis) > w:                      # truncate on visible text, keep codes balanced
        keep, out, n = w - 1, [], 0
        for part in re.split(r"(\033\[[0-9;]*m)", s):
            if ANSI.fullmatch(part):
                out.append(part); continue
            take = part[:max(0, keep - n)]
            out.append(take); n += len(take)
        return "|" + "".join(out) + ">" + C["0"] + "|"
    return "|" + s + " " * (w - len(vis)) + "|"


def bar(frac, w=28):
    frac = max(0.0, min(1.0, frac))
    n = int(round(frac * w))
    return "#" * n + "." * (w - n)


def render():
    done, cur = parse()
    now = time.time()
    procs = ps_python()
    trainer = next((p for p in procs if p["mb"] and p["mb"] > 300), None)
    fm, tm = free_mb()
    W = 66
    L = []
    L.append("+" + "-" * W + "+")
    title = "  AI Radiology Hub - training progress"
    L.append(row(C["B"] + title + C["0"] + " " * (W - len(title) - 9)
                 + time.strftime("%H:%M:%S") + " ", W))
    L.append("+" + "-" * W + "+")

    if cur is None:
        L.append(row("  no run in progress", W))
    else:
        ep, tot = cur["ep"], cur["total"]
        # rate: prefer stage-B epochs, they dominate the cost
        elapsed = (now - trainer["start"]) if (trainer and trainer["start"]) else None
        spe = eta = None
        if elapsed and ep > 0:
            b0 = cur["stageB_first"]
            if b0 and ep >= b0:
                # assume stage A epochs cost ~35% of a stage B epoch (frozen backbone)
                a_n = b0 - 1
                b_n = ep - a_n
                spe = elapsed / (b_n + 0.35 * a_n) if (b_n + 0.35 * a_n) > 0 else None
            else:
                spe = elapsed / ep
            if spe:
                eta = spe * (tot - ep)
        head = "  %s%s%s  %s %dpx  %d-class  [%s]" % (
            C["b"] + C["B"], cur["key"], C["0"], cur["dataset"], cur["size"],
            cur["classes"], cur["stage"])
        L.append(row(head, W))
        pct = (ep / tot) if tot else 0
        pl = "  [%s] %d/%d ep  %3d%%" % (bar(pct), ep, tot, int(pct * 100))
        L.append(row(pl, W))
        t1 = "  elapsed %s   remaining <= %s%s%s   %s/epoch" % (
            hms(elapsed), C["y"], hms(eta), C["0"], hms(spe))
        L.append(row(t1, W))
        if cur["val"] is not None:
            t2 = "  val_acc %.4f   best %.4f" % (cur["val"], cur["best"])
            L.append(row(t2, W))
        if trainer:
            t3 = "  pid %s   rss %.0f MB" % (trainer["pid"], trainer["mb"])
            L.append(row(t3, W))
        L.append(row("  note: ETA is an upper bound - early stopping can end it", W))

    if done:
        L.append("+" + "-" * W + "+")
        L.append(row("  recent runs", W))
        for d in done[-6:]:
            if d.get("state") == "fatal":
                s = "%s%-22s FATAL (non-finite loss)%s" % (C["r"], d["key"], C["0"])
            else:
                mark = C["g"] if d.get("result", 0) >= 0.85 else C["y"]
                s = "%-22s %stest %.4f%s   %d ep%s%s" % (
                    d["key"], mark, d.get("result", 0), C["0"], d["ep"],
                    "  (early stop)" if d.get("early_stopped") else "",
                    "  ~mixed-log" if d.get("interleaved") else "")
            L.append(row("  " + s, W))

    L.append("+" + "-" * W + "+")
    if fm:
        warn = C["r"] + " LOW" + C["0"] if fm < 900 else ""
        m = "  RAM free %.0f / %.0f MB%s" % (fm, tm, warn)
        L.append(row(m, W))
    if os.path.exists(LOG):
        age = now - os.path.getmtime(LOG)
        stale = C["r"] + "  (stale!)" + C["0"] if age > 600 else ""
        m = "  log updated %s ago%s" % (hms(age), stale)
        L.append(row(m, W))
    L.append("+" + "-" * W + "+")
    return "\n".join(L)


def main():
    watch = 0
    if len(sys.argv) > 1 and sys.argv[1] in ("-w", "--watch"):
        watch = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    if not watch:
        print(render()); return
    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            print(render())
            print("\n  refreshing every %ds - Ctrl+C to stop" % watch)
            time.sleep(watch)
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
