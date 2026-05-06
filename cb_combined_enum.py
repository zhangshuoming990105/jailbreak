"""Combined attack: AutoDAN-Turbo prompt + EnumAttack schema.

Less constrained variant of cb_combined.py: replaces DictAttack's
multi-stage dictionary/stage1/stage2 schema with EnumAttack v2 (prefix Enum
+ answer_body Enum + 5 free-form `steps`). The simpler schema leaves more
room for the AutoDAN scenario framing to influence the actual answer
content.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import json
from pathlib import Path
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from cb_eval import (
    ENUM_ATTACK_V2_TEMPLATE,
    enum_attack_schema_for,
    assemble_enum_output,
    load_autodan_turbo,
    REDTEAM_SYSTEM,
)


async def call_combined(base_url: str, model_id: str,
                        question: str, autodan_prompt: str,
                        max_tokens: int = 2048,
                        timeout: float = 120.0,
                        repetition_penalty: float = 1.3) -> str:
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)
    schema = enum_attack_schema_for(question)
    try:
        r = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": REDTEAM_SYSTEM},
                {"role": "user", "content": autodan_prompt},
            ],
            response_format=schema,
            temperature=0.6,
            max_tokens=max_tokens,
            extra_body={"repetition_penalty": repetition_penalty},
        )
        raw = r.choices[0].message.content or ""
        return assemble_enum_output(raw)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


async def run_target(name, base_url, model_id, args, outdir: Path):
    autodan = load_autodan_turbo(args.autodan_jsonl, args.begin, args.end)

    sem = asyncio.Semaphore(args.concurrency)

    async def _one(orig, ap):
        async with sem:
            return orig, await call_combined(base_url, model_id, orig, ap,
                                             max_tokens=args.max_tokens,
                                             timeout=args.timeout,
                                             repetition_penalty=args.repetition_penalty)

    logger.info(f"[{name}] running combined autodan+enum on {len(autodan)} questions")
    pairs = await asyncio.gather(*[_one(o, p) for o, p in autodan])
    csv_path = outdir / f"qa_advbench_combined_enum_{name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question", "answer"])
        for q, a in pairs:
            w.writerow([q, a])
    logger.info(f"[{name}] wrote {csv_path}")


async def main_async(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger.add(outdir / "cb_combined_enum.log", level="INFO")

    targets = []
    if args.targets in ("base", "both"):
        targets.append(("base", args.base_url, args.base_model_id))
    if args.targets in ("cb", "both"):
        targets.append(("cb", args.cb_url, args.cb_model_id))
    for name, url, mid in targets:
        await run_target(name, url, mid, args, outdir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8080/v1")
    p.add_argument("--cb-url", default="http://localhost:8090/v1")
    p.add_argument("--base-model-id", default="base-llama3-8b-instruct")
    p.add_argument("--cb-model-id", default="cb-llama3-8b-instruct-rr")
    p.add_argument("--targets", choices=["base", "cb", "both"], default="both")
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--repetition-penalty", type=float, default=1.3)
    p.add_argument("--autodan-jsonl",
                   default="autodan-turbo_advbench_best.jsonl")
    p.add_argument("--begin", type=int, default=0)
    p.add_argument("--end", type=int, default=50)
    p.add_argument("--outdir", default="cb_results")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
