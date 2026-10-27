"""RL on Qwen3.5-9B over the patent practice matters (user, 2026-10-27: Sonnet judges the reward).

Each round: 32 training matters x 4 rollouts on Modal (temperature 0.8); each rollout written as its deliverable and
graded locally (12 at once) by one Sonnet judge over the matter's full rubric (reward = share of criteria met, 0 when the quotation
gate fails); group-normalised advantages; one policy-gradient pass on Modal that saves the LoRA and a merged copy.
After the last round the merged model answers the 22 test prompts (greedy, as before) -> out_after_rl.jsonl, graded
with Opus by grade.py like every other arm. Resumable: rollouts and rewards of a round are kept on disk.

  LEGALRL_AUTH=subscription .venv/bin/python scripts/sft_fto/rl.py [rounds]   (3 -> out_after_rl.jsonl, else out_after_rl<n>.jsonl)
  LEGALRL_AUTH=subscription .venv/bin/python scripts/sft_fto/grade.py after_rl[<n>]

Rounds 1-3 used lr 2e-5; later rounds continue from them at 1e-4 (user, 2026-10-27: a longer run). Before each new
round the Modal bill is read; at BUDGET_STOP dollars no new round starts and the latest model is evaluated.
"""
import json
import random
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import modal_9b as m
from grade import write_docx

from legalrl.world.grade import file_texts, grade_task

D = m.LOCAL
RL = D / "rl"
WP = D.parent.parent / "data/bench/worlds/patent_practice"
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
PER_ROUND, N = 32, 4
BUDGET_STOP = 27.5   # dollars of the $30 Modal credit; an eval pass after the stop costs about $0.3
JUDGES = 12   # Sonnet reward calls at once (user, 2026-10-27; was 6)


def reward(rid: str, k: int, text: str, rnd: int) -> float:
    w = WP / rid
    task = json.loads((w / "hidden/tasks.json").read_text())[0]
    od = RL / f"r{rnd}" / f"{rid}_{k}" / "output"
    write_docx(text, od / Path(task["deliverable"]).name)
    g = grade_task(w, task["id"], od, {}, w / "hidden" / "judge_cache", "sonnet", file_texts(w), judge_mode="full_rubric")
    return float(g.get("dense") or 0.0)


def modal_spent() -> float:
    out = subprocess.run([str(D.parent.parent / ".venv/bin/modal"), "billing", "summary"], capture_output=True, text=True, check=False).stdout
    m_ = re.search(r"Metered Cost:\s+([\d.]+)", out)
    return float(m_.group(1)) if m_ else 0.0


def main() -> None:
    RL.mkdir(parents=True, exist_ok=True)
    rows = {r["id"]: r for r in map(json.loads, (D / "train_v2.jsonl").read_text().splitlines())}
    order = sorted(rows)
    random.Random(0).shuffle(order)
    order = order * 4
    adapter, path, last = None, m.MODEL, -1
    with m.app.run():
        for rnd in range(ROUNDS):
            tag = f"r{rnd}"
            if (RL / f"{tag}.done").exists():
                adapter, path, last = f"{m.CACHE}/rl/{tag}/adapter", f"{m.CACHE}/rl/{tag}/merged", rnd
                continue
            spent = modal_spent()
            if spent >= BUDGET_STOP:
                print(json.dumps({"stopped_before_round": rnd, "modal_spent": spent}), flush=True)
                break
            batch = [rows[i] for i in order[rnd * PER_ROUND:(rnd + 1) * PER_ROUND]]
            ro_f = RL / f"{tag}_rollouts.jsonl"
            if not ro_f.exists():
                ro = m.rollouts.remote(batch, path, N)
                ro_f.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in ro))
            ro = [json.loads(x) for x in ro_f.read_text().splitlines()]
            rw_f = RL / f"{tag}_rewards.jsonl"
            done = {(x["id"], x["k"]): x["reward"] for x in map(json.loads, rw_f.read_text().splitlines())} if rw_f.exists() else {}
            todo = [x for x in ro if (x["id"], x["k"]) not in done]

            def one(x, rnd=rnd, rw_f=rw_f, done=done, tag=tag):
                try:
                    r = reward(x["id"], x["k"], x["text"], rnd)
                except Exception as e:  # noqa: BLE001 - a failed judge call leaves the rollout out of the step
                    print("reward failed", x["id"], x["k"], e, flush=True)
                    return
                with rw_f.open("a") as f:
                    f.write(json.dumps({"id": x["id"], "k": x["k"], "reward": r, "finish": x["finish"]}) + "\n")
                done[(x["id"], x["k"])] = r
                print(tag, x["id"][16:56], x["k"], round(r, 3), flush=True)

            with ThreadPoolExecutor(JUDGES) as ex:
                list(ex.map(one, todo))
            items = []
            for rid in {x["id"] for x in ro}:
                g = [x for x in ro if x["id"] == rid and (rid, x["k"]) in done]
                rs = [done[(rid, x["k"])] for x in g]
                if len(rs) < 2 or statistics.pstdev(rs) < 1e-6:
                    continue
                mu, sd = statistics.mean(rs), statistics.pstdev(rs)
                items += [{"messages": rows[rid]["messages"][:2], "text": x["text"], "adv": (r - mu) / (sd + 1e-6)} for x, r in zip(g, rs)]
            print(json.dumps({"round": rnd, "rollouts": len(done), "mean_reward": round(statistics.mean(done.values()), 4),
                              "items": len(items)}), flush=True)
            res = m.pg_step.remote(items, adapter, tag, 2e-5 if rnd < 3 else 1e-4)
            (RL / f"{tag}_step.json").write_text(json.dumps(res))
            (RL / f"{tag}.done").write_text("")
            adapter, path = f"{m.CACHE}/rl/{tag}/adapter", f"{m.CACHE}/rl/{tag}/merged"
            last = rnd
        test = [json.loads(x) for x in (D / "test.jsonl").read_text().splitlines()]
        out = m.generate.remote(test, path)
        name = "out_after_rl.jsonl" if last == 2 else f"out_after_rl{last + 1}.jsonl"
        (D / name).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    print(f"eval outputs written ({name}); grade with: grade.py {name[4:-6]}", flush=True)


if __name__ == "__main__":
    main()
