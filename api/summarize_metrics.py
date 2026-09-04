# -*- coding: utf-8 -*-
"""
Reads every models/**/*_metrics.json and prints one table of measured accuracies, flagging
anything under the 91% bar. This is the script behind the audit table in TRAINING_LOG.md —
so the table can be regenerated instead of hand-maintained (and can't quietly go stale).

  python summarize_metrics.py            # human table
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
            "extra": extra,
        })
    return rows


def main():
    rows = collect()
    scored = [r for r in rows if r["accuracy"] is not None]
    scored.sort(key=lambda r: -r["accuracy"])
    unscored = [r for r in rows if r["accuracy"] is None]

    if "--json" in sys.argv:
        print(json.dumps({"rows": rows, "bar": BAR}, ensure_ascii=False, indent=2)); return

    md = "--md" in sys.argv
    if md:
        print("| file | task | test acc | train acc | gap | AUC | n_test | status |")
        print("|---|---|---|---|---|---|---|---|")
        for r in scored:
            print("| `%s` | %s | **%.2f%%** | %s | %s | %s | %s | %s |" % (
                r["file"], r["task"], r["accuracy"] * 100,
                "%.2f%%" % (r["train_accuracy"] * 100) if r["train_accuracy"] else "—",
                "%+.4f" % r["gap"] if r["gap"] is not None else "—",
                r["auc"] if r["auc"] else "—", r["n_test"] or "—",
                "✅" if r["accuracy"] >= BAR else "❌ under 91%"))
        for r in unscored:
            print("| `%s` | %s | — | — | — | — | — | %s |" % (r["file"], r["task"], r["extra"] or "n/a"))
        return

    print("%-46s %8s  %8s  %8s  %s" % ("metrics file", "test", "train", "gap", "task"))
    print("-" * 110)
    for r in scored:
        flag = "  " if r["accuracy"] >= BAR else " <"
        print("%-46s %7.2f%%%s %8s  %8s  %s" % (
            r["file"][:46], r["accuracy"] * 100, flag,
            "%.2f%%" % (r["train_accuracy"] * 100) if r["train_accuracy"] else "-",
            "%+.4f" % r["gap"] if r["gap"] is not None else "-", r["task"][:44]))
    for r in unscored:
        print("%-46s %8s  %8s  %8s  %s %s" % (r["file"][:46], "n/a", "-", "-", r["task"][:44], r["extra"]))
    under = [r for r in scored if r["accuracy"] < BAR]
    print("-" * 110)
    print("%d scored models | %d at or above %.0f%% | %d below: %s"
          % (len(scored), len(scored) - len(under), BAR * 100, len(under),
             ", ".join(r["file"].replace("_metrics.json", "") for r in under) or "none"))


if __name__ == "__main__":
    main()
