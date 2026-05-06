"""
IdeaGen - End-to-End Autonomous Research Hypothesis Discovery Pipeline

Flow:
  Phase 1  Deep Research  →  search papers, synthesize background
  Phase 2  Hypothesis Gen →  multi-agent: generate, reflect, evolve, rank, develop
  Phase 3  Report         →  assemble Markdown research report
"""

import asyncio
import json
import logging
import os
import os.path as osp
import sys
import time
import yaml
import argparse
from datetime import datetime

from dotenv import load_dotenv

from internagent.deep_research import DeepResearcher
from internagent.stage import IdeaGenerator
from internagent.report_generator import ReportGenerator
from internagent.report_pdf import generate_report_pdf

load_dotenv()


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    log_file = osp.join("logs", f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_ideagen.log')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    return logging.getLogger("IdeaGen")


# ── Arguments ─────────────────────────────────────────────────────────────────

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IdeaGen: Autonomous Deep Research + Hypothesis Generation"
    )

    # ── Core inputs
    core = parser.add_argument_group("Research Input")
    core.add_argument("--topic", type=str, required=True,
                      help="Research topic (e.g. 'Protein structure prediction')")
    core.add_argument("--domain", type=str, required=True,
                      help="Research domain (e.g. 'computational biology')")
    core.add_argument("--background", type=str, default="",
                      help="Optional extra background context")
    core.add_argument("--constraints", type=str, nargs="*", default=[],
                      help="Research constraints (e.g. 'must run on single GPU')")

    # ── Pipeline control
    pipe = parser.add_argument_group("Pipeline Control")
    pipe.add_argument("--skip_deep_research", action="store_true",
                      help="Skip Phase 1 and load existing deep_research.json")
    pipe.add_argument("--skip_idea_generation", action="store_true",
                      help="Skip Phase 2 and load existing ideas.json")

    # ── Config & output
    cfg = parser.add_argument_group("Config & Output")
    cfg.add_argument("--config", type=str, default="config/default_config.yaml",
                     help="Path to YAML configuration file")
    cfg.add_argument("--offline_feedback", type=str,
                     default="config/feedback_global.json",
                     help="Offline feedback JSON for hypothesis generation")
    cfg.add_argument("--output_dir", type=str, default=None,
                     help="Output directory (default: results/<sanitized_topic>)")

    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def sanitize_name(text: str, max_len: int = 40) -> str:
    """Turn topic string into a safe directory name."""
    safe = text.strip().replace(" ", "_")
    safe = "".join(c for c in safe if c.isalnum() or c in "_-")
    return safe[:max_len]


def load_config(path: str) -> dict:
    if path and osp.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def banner(logger: logging.Logger, title: str) -> None:
    logger.info("=" * 65)
    logger.info(f"  {title}")
    logger.info("=" * 65)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger = setup_logging()
    args = parse_arguments()
    config = load_config(args.config)

    # Output directory
    task_name = sanitize_name(args.topic)
    args.output_dir = args.output_dir or osp.join("results", task_name)
    os.makedirs(args.output_dir, exist_ok=True)

    pipeline_start = time.time()
    phase_times = {}

    banner(logger, "IdeaGen Pipeline Started")
    logger.info(f"  Topic      : {args.topic}")
    logger.info(f"  Domain     : {args.domain}")
    logger.info(f"  Output Dir : {args.output_dir}")

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 1 · Deep Research
    # ──────────────────────────────────────────────────────────────────────────
    dr_path = osp.join(args.output_dir, "deep_research.json")

    if args.skip_deep_research and osp.exists(dr_path):
        logger.info("Skipping Deep Research – loading existing results...")
        with open(dr_path, "r", encoding="utf-8") as f:
            deep_research_result = json.load(f)
    else:
        banner(logger, "Phase 1 · Deep Research")
        t1 = time.time()
        researcher = DeepResearcher(
            topic=args.topic,
            domain=args.domain,
            config=config,
            background=args.background,
        )
        deep_research_result = asyncio.run(researcher.run())
        phase_times["Phase 1 · Deep Research"] = time.time() - t1

        with open(dr_path, "w", encoding="utf-8") as f:
            json.dump(deep_research_result, f, ensure_ascii=False, indent=2)
        logger.info(
            f"Deep research complete – {deep_research_result['paper_count']} papers found "
            f"({phase_times['Phase 1 · Deep Research']:.1f}s)"
        )
        logger.info(f"  Saved : {dr_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 2 · Hypothesis Generation
    # ──────────────────────────────────────────────────────────────────────────
    ideas_path = osp.join(args.output_dir, "ideas.json")

    if args.skip_idea_generation and osp.exists(ideas_path):
        logger.info("Skipping Hypothesis Generation – loading existing ideas...")
        with open(ideas_path, "r", encoding="utf-8") as f:
            top_ideas = json.load(f)
    else:
        banner(logger, "Phase 2 · Hypothesis Generation (Multi-Agent)")

        # Write a prompt.json dynamically so IdeaGenerator can load it
        # Inject the deep-research synthesis as the background context
        prompt_data = {
            "task_description": args.topic,
            "domain": args.domain,
            "background": deep_research_result.get("synthesis", ""),
            "constraints": args.constraints,
        }
        prompt_path = osp.join(args.output_dir, "prompt.json")
        with open(prompt_path, "w", encoding="utf-8") as f:
            json.dump(prompt_data, f, ensure_ascii=False, indent=2)

        # Attach necessary fields to args for IdeaGenerator
        args.task_name = task_name
        args.task_dir = args.output_dir   # prompt.json lives here
        args.ref_code_path = None         # no experiment code needed

        idea_generator = IdeaGenerator(args, logger)
        t2 = time.time()
        try:
            top_ideas, session_json = asyncio.run(idea_generator.generate_ideas())
        except Exception as exc:
            logger.error(f"Hypothesis generation failed: {exc}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        phase_times["Phase 2 · Hypothesis Gen"] = time.time() - t2

        with open(ideas_path, "w", encoding="utf-8") as f:
            json.dump(top_ideas, f, ensure_ascii=False, indent=2)
        logger.info(f"  {len(top_ideas)} hypotheses saved ({phase_times['Phase 2 · Hypothesis Gen']:.1f}s) : {ideas_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 3 · Report Generation
    # ──────────────────────────────────────────────────────────────────────────
    banner(logger, "Phase 3 · Report Generation")
    t3 = time.time()

    report_gen = ReportGenerator()
    report_md = report_gen.generate(
        topic=args.topic,
        domain=args.domain,
        deep_research=deep_research_result,
        top_ideas=top_ideas,
    )
    report_path = osp.join(args.output_dir, "report.md")
    report_gen.save(report_md, report_path)
    logger.info(f"  Report saved : {report_path}")

    # Generate readable PDF report
    report_pdf_path = osp.join(args.output_dir, "report.pdf")
    try:
        generate_report_pdf(
            topic=args.topic,
            domain=args.domain,
            deep_research=deep_research_result,
            top_ideas=top_ideas,
            output_path=report_pdf_path,
        )
        logger.info(f"  PDF report saved : {report_pdf_path}")
    except Exception as exc:
        logger.warning(f"  PDF report generation failed: {exc} (Markdown report is still available)")

    # ──────────────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────────────
    phase_times["Phase 3 · Report"] = time.time() - t3
    total_time = time.time() - pipeline_start

    banner(logger, "All Done!")
    logger.info(f"  Papers found       : {deep_research_result.get('paper_count', 0)}")
    logger.info(f"  Hypotheses         : {len(top_ideas)}")
    logger.info(f"  Output directory   : {args.output_dir}")
    logger.info(f"  Report             : {report_path}")
    logger.info("")
    logger.info("  ⏱ Timing Breakdown:")
    for phase_name, elapsed in phase_times.items():
        mins, secs = divmod(elapsed, 60)
        logger.info(f"    {phase_name:30s} : {int(mins):02d}:{secs:05.2f}")
    total_mins, total_secs = divmod(total_time, 60)
    logger.info(f"    {'TOTAL':30s} : {int(total_mins):02d}:{total_secs:05.2f}")

    print("\n📋 Top Hypotheses:")
    for i, idea in enumerate(top_ideas, 1):
        details = idea.get("refined_method_details") or idea.get("method_details") or {}
        title = details.get("title") or idea.get("text", "N/A")
        score = idea.get("score", 0)
        print(f"  {i}. [{score:.2f}] {title}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
    except Exception as exc:
        print(f"\nFatal error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
