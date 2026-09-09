# -*- coding: utf-8 -*-
"""
Reads every models/**/*_metrics.json and prints one table of measured accuracies, flagging
anything under the 91% bar. This is the script behind the audit table in TRAINING_LOG.md —
so the table can be regenerated instead of hand-maintained (and can't quietly go stale).

  python summarize_metrics.py            # human table of EVERY metrics file
  python summarize_metrics.py --served   # only what main.py would actually load
  python summarize_metrics.py --md       # markdown table
  python summarize_metrics.py --json     # machine readable
"""
import json, os, sys, glob

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
BAR = 0.91


def collect():
    rows = []
    # The MARBERT specialty router keeps its eval in router_meta.json, not a *_metrics.json.
    # It is the router ar_service.py ACTUALLY serves whenever `transformers` imports, so an
    # audit that only globs *metrics*.json reports the superseded linear router's score and
    # understates the live system. Pick it up explicitly.
    paths = sorted(glob.glob(os.path.join(MODEL_DIR, "**", "*metrics*.json"), recursive=True))
    paths += sorted(glob.glob(os.path.join(MODEL_DIR, "**", "router_meta.json"), recursive=True))
    for path in paths:
        name = os.path.relpath(path, MODEL_DIR).replace("\\", "/")
        try:
            with open(path, encoding="utf-8") as f:
                m = json.load(f)
        except Exception as e:
            rows.append({"file": name, "task": "UNREADABLE (%s)" % e, "accuracy": None}); continue
        if not isinstance(m, dict):
            continue

        # Image / single-label models report test_accuracy directly.
        acc = m.get("test_accuracy")
        task = m.get("task") or m.get("model") or name
        extra = ""
        if acc is None:
            # Text systems and the multi-label chest model use different headline metrics.
            if "end_to_end_accuracy" in m:                       # English symptom system
                acc, task = m["end_to_end_accuracy"], "text: symptom -> diagnosis (end-to-end)"
            elif "router" in m and isinstance(m["router"], dict):  # Arabic system
                r = m["router"]
                acc = r.get("accuracy")
                task = "arabic specialty router (20-way)"
                extra = "top-3 %.4f" % r["top3_accuracy"] if r.get("top3_accuracy") else ""
            elif "mean_auc" in m:                                 # chest, multi-label
                task = "chest 14-label (AUC only - accuracy is not defined for multi-label)"
                extra = "mean AUC %.4f" % m["mean_auc"]
            elif "eval" in m and isinstance(m["eval"], dict):      # MARBERT router_meta.json
                e = m["eval"]
                acc = e.get("top1")
                task = "arabic specialty router (MARBERT, 20-way) [SERVED]"
                extra = "top-3 %.4f" % e["top3"] if e.get("top3") else ""
                m = {"train_accuracy": e.get("train_acc"), "overfitting_gap": e.get("gap")}
            elif "accuracy" in m:                                 # arabic input filter
                acc = m["accuracy"]

        rows.append({
            "file": name, "task": task, "accuracy": acc,
            "train_accuracy": m.get("train_accuracy"),
            "gap": m.get("overfitting_gap"),
            "auc": m.get("test_auc"),
            "n_test": m.get("n_test"),
            "trained_at": m.get("trained_at") or m.get("validated_at"),
            "arch": m.get("arch"),
            # Session 6 found the TTA flag was being picked by scoring the TEST split twice
            # and keeping the winner, which can only move the headline up. A number selected
            # that way is not comparable to one selected on val, so the inventory says which
            # is which instead of printing them side by side as if they were equivalent.
            "tta_on_test": ("tta_used" in m and m.get("tta_decided_on") != "val"),
            "extra": extra,
        })
    return rows


# Resolution order copied from main.py:460-500. A manifest wins, then the v2 retrain, then
# the v1 checkpoint. Session 1 step 7 already recorded what happens without this: an audit
# that globs metrics files reported the superseded linear Arabic router instead of the served
# MARBERT one, and understated it by 5 points. The same trap caught a hand-written table on
# 2026-09-08, which picked up retina_ens_metrics.json (0.635, the abandoned session-2
# ensemble) instead of the served retina_v2 (0.6700).
SERVED_KEYS = ["breast", "derma", "derma_bin", "blood", "organc", "path",
               "oct", "oct_bin", "retina", "retina_bin"]


def served_metrics_file(key):
    """The metrics file describing what main.py would serve for `key`, or None."""
    man = os.path.join(MODEL_DIR, key + "_ensemble.json")
    if os.path.exists(man):
        try:
            with open(man, encoding="utf-8") as f:
                name = json.load(f).get("metrics", key + "_ens_metrics.json")
            if os.path.exists(os.path.join(MODEL_DIR, name)):
                return name, "ensemble"
        except Exception:
            pass
    if os.path.exists(os.path.join(MODEL_DIR, key + "_v2.pt")):
        return key + "_v2_metrics.json", "v2"
    return key + "_metrics.json", "v1"


def served_rows():
    out = []
    for key in SERVED_KEYS:
        name, kind = served_metrics_file(key)
        p = os.path.join(MODEL_DIR, name)
        if not os.path.exists(p):
            out.append({"key": key, "file": name, "kind": kind, "accuracy": None})
            continue
        with open(p, encoding="utf-8") as f:
            m = json.load(f)
        out.append({"key": key, "file": name, "kind": kind,
                    "accuracy": m.get("test_accuracy"), "arch": m.get("arch", "resnet18"),
                    "caught": m.get("disease_caught"), "members": m.get("members"),
                    "gap": m.get("overfitting_gap"), "auc": m.get("test_auc")})
    return out


def print_served():
    rows = served_rows()
    print("%-12s %-9s %-18s %-12s %-9s %s"
          % ("served id", "accuracy", "arch", "caught", "source", "note"))
    print("-" * 84)
    for r in sorted(rows, key=lambda r: -(r["accuracy"] or 0)):
        note = ("ensemble of %d" % len(r["members"])) if r.get("members") else ""
        if r["accuracy"] is not None and r["accuracy"] < BAR:
            note = (note + "  ") if note else ""
            note += "< 91%"
        print("%-12s %-9s %-18s %-12s %-9s %s"
              % (r["key"], r["accuracy"], r.get("arch", "-"), r.get("caught") or "",
                 r["kind"], note))
    scored = [r for r in rows if r["accuracy"] is not None]
    over = [r for r in scored if r["accuracy"] >= BAR]
    print("-" * 84)
    print("%d of %d served image models at or above %.0f%%" % (len(over), len(scored), BAR * 100))


def main():
    if "--served" in sys.argv:
        print_served()
        return
    rows = collect()
    scored = [r for r in rows if r["accuracy"] is not None]
    scored.sort(key=lambda r: -r["accuracy"])
    unscored = [r for r in rows if r["accuracy"] is None]

    if "--json" in sys.argv:
        print(json.dumps({"rows": rows, "bar": BAR}, ensure_ascii=False, indent=2)); return

    md = "--md" in sys.argv
    if md:
        print("| file | task | arch | test acc | train acc | gap | AUC | n_test | status |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in scored:
            st = "✅" if r["accuracy"] >= BAR else "❌ under 91%"
            if r["tta_on_test"]:
                st += " ⚠ TTA picked on test"
            print("| `%s` | %s | %s | **%.2f%%** | %s | %s | %s | %s | %s |" % (
                r["file"], r["task"], r["arch"] or "—", r["accuracy"] * 100,
                "%.2f%%" % (r["train_accuracy"] * 100) if r["train_accuracy"] else "—",
                "%+.4f" % r["gap"] if r["gap"] is not None else "—",
                r["auc"] if r["auc"] else "—", r["n_test"] or "—", st))
        for r in unscored:
            print("| `%s` | %s | — | — | — | — | — | %s |" % (r["file"], r["task"], r["extra"] or "n/a"))
        return

    print("%-40s %-16s %8s  %8s  %8s  %s" % ("metrics file", "arch", "test", "train", "gap", "task"))
    print("-" * 118)
    for r in scored:
        flag = "  " if r["accuracy"] >= BAR else " <"
        print("%-40s %-16s %7.2f%%%s %8s  %8s  %s%s" % (
            r["file"][:40], (r["arch"] or "-")[:16], r["accuracy"] * 100, flag,
            "%.2f%%" % (r["train_accuracy"] * 100) if r["train_accuracy"] else "-",
            "%+.4f" % r["gap"] if r["gap"] is not None else "-", r["task"][:38],
            "  [!] TTA picked on test" if r["tta_on_test"] else ""))
    for r in unscored:
        print("%-40s %-16s %8s  %8s  %8s  %s %s" % (
            r["file"][:40], "-", "n/a", "-", "-", r["task"][:38], r["extra"]))
    under = [r for r in scored if r["accuracy"] < BAR]
    print("-" * 110)
    print("%d scored models | %d at or above %.0f%% | %d below: %s"
          % (len(scored), len(scored) - len(under), BAR * 100, len(under),
             ", ".join(r["file"].replace("_metrics.json", "") for r in under) or "none"))
    tainted = [r for r in scored if r["tta_on_test"]]
    if tainted:
        print("[!] %d model(s) still have a TTA flag chosen on the test split - run "
              "fix_tta_selection.py --write: %s"
              % (len(tainted), ", ".join(r["file"].replace("_metrics.json", "") for r in tainted)))


if __name__ == "__main__":
    main()
