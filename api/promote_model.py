# -*- coding: utf-8 -*-
"""
Promote an experiment checkpoint to the name main.py actually serves - reversibly, and with
the reason recorded next to the weights.

main.py loads `<key>_v2.pt` when it exists and falls back to `<key>.pt` (main.py:467-469), so
"promoting" means writing the experiment's weights to `<key>_v2.pt`. For retina that OVERWRITES
a served file. Doing that by hand with `copy` leaves no way back and no record of why, which is
the opposite of how every other decision in this project is handled.

What this does instead:
  1. Refuses to promote a checkpoint whose metrics file does not exist, or whose TTA flag was
     still selected on test (session 6, step 38) - that number was never eligible.
  2. Refuses when val does not agree with the promotion, unless --force-val-disagrees is given
     with a written reason. On breast, val ranks the three candidates in exactly the reverse
     order of test (step 45); promoting there on a test comparison is test-set selection.
  3. Backs the current served checkpoint and metrics up to models/_promoted/<key>/<timestamp>/
     before touching anything.
  4. Writes models/_promotions.json: what moved, from what, when, why, and both numbers.

    python promote_model.py --from derma_bin_eb0 --to derma_bin --why "..."
    python promote_model.py --from derma_bin_eb0 --to derma_bin --why "..." --apply
"""
import os, json, time, shutil, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
BACKUP_DIR = os.path.join(MODEL_DIR, "_promoted")
LEDGER = os.path.join(MODEL_DIR, "_promotions.json")


def load_metrics(key):
    p = os.path.join(MODEL_DIR, key + "_metrics.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def best_val(met):
    h = met.get("epoch_history") or []
    return max((e["val_acc"] for e in h), default=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True, help="experiment key, e.g. derma_bin_eb0")
    ap.add_argument("--to", dest="dst", required=True, help="served key, e.g. derma_bin")
    ap.add_argument("--why", required=True, help="one sentence, recorded in the ledger")
    ap.add_argument("--apply", action="store_true", help="actually write (default: dry run)")
    ap.add_argument("--force-val-disagrees", action="store_true",
                    help="promote even though val ranks the incumbent higher")
    args = ap.parse_args()

    src_ck = os.path.join(MODEL_DIR, args.src + ".pt")
    src_met = load_metrics(args.src)
    if not os.path.exists(src_ck) or src_met is None:
        raise SystemExit("no checkpoint or metrics for %s" % args.src)

    # The incumbent is whatever main.py would load today.
    v2 = os.path.join(MODEL_DIR, args.dst + "_v2.pt")
    incumbent_key = (args.dst + "_v2") if os.path.exists(v2) else args.dst
    inc_met = load_metrics(incumbent_key)
    dst_ck = v2                     # promotion always writes the _v2 name main.py prefers

    print("promote  %s  ->  %s" % (args.src, os.path.basename(dst_ck)))
    print("incumbent: %s" % incumbent_key)

    # --- gate 1: the number has to have been eligible in the first place --------------------
    if src_met.get("tta_decided_on") != "val" and "tta_used" in src_met:
        raise SystemExit("REFUSED: %s still has its TTA flag chosen on the test split. "
                         "Run fix_tta_selection.py --write %s first." % (args.src, args.src))

    s_acc, i_acc = src_met.get("test_accuracy"), (inc_met or {}).get("test_accuracy")
    s_val, i_val = best_val(src_met), best_val(inc_met or {})
    print("  test : %-8s -> %-8s" % (i_acc, s_acc))
    print("  val  : %-8s -> %-8s" % (i_val, s_val))

    # --- gate 2: val has to agree -----------------------------------------------------------
    if s_val is not None and i_val is not None and s_val <= i_val:
        msg = ("REFUSED: val ranks the incumbent higher (%.4f vs %.4f). Test says otherwise, "
               "but choosing on test is the bug this project fixed in step 38. Pass "
               "--force-val-disagrees with a reason if there is a non-test argument."
               % (i_val, s_val))
        if not args.force_val_disagrees:
            raise SystemExit(msg)
        print("  [forced] " + msg)

    # --- clinical guard: never promote something that catches fewer cases -------------------
    def caught(met):
        cm = met.get("confusion_matrix")
        if not cm or len(cm) != 2:
            return None
        pos = 1 if met.get("binary_task") else (0 if met.get("medmnist") == "breastmnist" else 1)
        return cm[pos][pos], sum(cm[pos])

    c_s, c_i = caught(src_met), caught(inc_met or {})
    if c_s and c_i:
        print("  disease caught: %d/%d -> %d/%d" % (c_i[0], c_i[1], c_s[0], c_s[1]))
        if c_s[0] < c_i[0]:
            raise SystemExit("REFUSED: catches %d cases against the incumbent's %d. Accuracy "
                             "bought with missed disease is not an improvement (step 14)."
                             % (c_s[0], c_i[0]))

    if not args.apply:
        print("\ndry run - nothing written. Re-run with --apply.")
        return

    # --- back up whatever is being replaced --------------------------------------------------
    stamp = time.strftime("%Y%m%d-%H%M%S")
    bdir = os.path.join(BACKUP_DIR, args.dst, stamp)
    os.makedirs(bdir, exist_ok=True)
    for p in (dst_ck, os.path.join(MODEL_DIR, incumbent_key + "_metrics.json")):
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(bdir, os.path.basename(p)))
            print("  backed up %s -> %s" % (os.path.basename(p), bdir))

    shutil.copy2(src_ck, dst_ck)
    out_met = os.path.join(MODEL_DIR, args.dst + "_v2_metrics.json")
    met = dict(src_met)
    met["promoted_from"] = args.src
    met["promoted_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    met["promotion_reason"] = args.why
    met["replaced"] = {"key": incumbent_key, "test_accuracy": i_acc, "backup": bdir}
    with open(out_met, "w", encoding="utf-8") as f:
        json.dump(met, f, ensure_ascii=False, indent=2)
    print("  wrote %s and %s" % (os.path.basename(dst_ck), os.path.basename(out_met)))

    ledger = []
    if os.path.exists(LEDGER):
        with open(LEDGER, encoding="utf-8") as f:
            ledger = json.load(f)
    ledger.append({
        "at": met["promoted_at"], "from": args.src, "to": args.dst,
        "from_test_accuracy": s_acc, "replaced_test_accuracy": i_acc,
        "from_best_val": s_val, "replaced_best_val": i_val,
        "arch": src_met.get("arch", "resnet18"), "why": args.why,
        "val_disagreed": bool(args.force_val_disagrees), "backup": bdir,
    })
    with open(LEDGER, "w", encoding="utf-8") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=2)
    print("  ledger: %s (%d entries)" % (os.path.basename(LEDGER), len(ledger)))
    print("\nrestart the API to serve it.")


if __name__ == "__main__":
    main()
