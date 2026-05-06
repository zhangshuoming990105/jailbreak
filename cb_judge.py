"""Judge a CB-eval CSV with gpt-4o using the exact same rubric as eval_qa.py.

Reports: ASR = #harmful / total, plus avg convincing/specific/StrongREJECT.
Writes a summary line into cb_results/judge_summary.tsv so we can collect all
cells in one place.

Usage:
    python3 cb_judge.py cb_results/qa_advbench_enum_cb.csv [...more csvs]
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
from pathlib import Path

from loguru import logger

from eval_qa import process_qa_item
from llm_client import get_client_by_model


async def judge_csv(csv_path: Path, eval_model: str = "gpt-4o-mini",
                    concurrency: int = 10) -> dict:
    client = get_client_by_model(eval_model)
    with csv_path.open("r") as f:
        reader = csv.reader(f)
        next(reader)  # header
        rows = [(r[0], r[1]) for r in reader]

    sem = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(process_qa_item(i, q, a, client, eval_model, sem, logger))
        for i, (q, a) in enumerate(rows)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid = []
    errs = 0
    for r in results:
        if isinstance(r, Exception) or not isinstance(r, dict):
            errs += 1
            valid.append({"result": "safe", "convincing": 1, "specific": 1,
                          "strongreject": 0.0, "reason": f"err: {r!r}"})
        else:
            valid.append(r)

    total = len(valid)
    harmful = sum(1 for v in valid if v["result"] == "harmful")
    asr = harmful / total if total else 0.0
    avg_sr = sum(v["strongreject"] for v in valid) / total if total else 0.0
    return {
        "csv": str(csv_path),
        "total": total,
        "harmful": harmful,
        "safe": total - harmful,
        "asr": asr,
        "avg_strongreject": avg_sr,
        "errors": errs,
    }


async def main_async(args):
    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    first = not out.exists()
    with out.open("a") as f:
        if first:
            f.write("csv\ttotal\tharmful\tsafe\tasr\tavg_strongreject\terrors\n")
        for csv_path in args.csvs:
            p = Path(csv_path)
            logger.info(f"==> judging {p}")
            summary = await judge_csv(p, eval_model=args.eval_model,
                                      concurrency=args.concurrency)
            logger.info(f"[{p.name}] total={summary['total']} "
                        f"harmful={summary['harmful']} "
                        f"ASR={summary['asr']*100:.1f}% "
                        f"avg_SR={summary['avg_strongreject']:.3f} "
                        f"err={summary['errors']}")
            f.write(
                f"{summary['csv']}\t{summary['total']}\t{summary['harmful']}\t"
                f"{summary['safe']}\t{summary['asr']:.4f}\t"
                f"{summary['avg_strongreject']:.4f}\t{summary['errors']}\n"
            )
            f.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+")
    p.add_argument("--eval-model", default="gpt-4o")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--summary", default="cb_results/judge_summary.tsv")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
