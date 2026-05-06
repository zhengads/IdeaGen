"""
Research Report Generator for IdeaGen

Synthesizes Deep Research findings and top hypotheses
into a well-structured Markdown research report.
"""

from datetime import datetime
from typing import Any, Dict, List
import re  # 仅新增导入


class ReportGenerator:
    """
    Generates a structured Markdown research report from:
      - Deep research context (papers + synthesis)
      - Top hypotheses produced by the MAS pipeline
    """

    # 新增：文本清洗函数（过滤脏数据，无侵入）
    def _clean_text(self, text: str) -> str:
        if not text:
            return "N/A"
        text = str(text)
        text = re.sub(r'Page\s*\d+/\d+', '', text)  # 删分页乱码
        text = re.sub(r'[￬Λ|]', '', text)          # 删乱码符号
        text = re.sub(r'\s+', ' ', text)           # 合并多余空格
        return text.strip()

    def generate(
        self,
        topic: str,
        domain: str,
        deep_research: Dict[str, Any],
        top_ideas: List[Dict[str, Any]],
    ) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        paper_count = deep_research.get("paper_count", 0)
        papers = deep_research.get("papers", [])
        synthesis = self._clean_text(deep_research.get("synthesis", "N/A"))  # 清洗

        sections: List[str] = []

        # ── Header ──────────────────────────────────────────────────────────
        sections += [
            f"# Research Report: {topic}",
            "",
            f"> **Domain**: {domain}  |  "
            f"**Papers Surveyed**: {paper_count}  |  "
            f"**Hypotheses Generated**: {len(top_ideas)}  |  "
            f"**Generated**: {now}",
            "",
            "---",
            "",
        ]

        # ── Section 1: Research Background ──────────────────────────────────
        sections += [
            "## 1. Research Background & Literature Synthesis",
            "",
            synthesis,
            "",
            "---",
            "",
        ]

        # ── Section 2: Key Papers ────────────────────────────────────────────
        sections += [
            "## 2. Key References",
            "",
            f"The following papers were identified across arXiv and Semantic Scholar "
            f"(top {min(12, len(papers))} shown):",
            "",
        ]

        top_papers = papers[:12]
        if top_papers:
            sections.append("| # | Title | Year | Source |")
            sections.append("|---|-------|------|--------|")
            for i, p in enumerate(top_papers, 1):
                title = self._truncate(self._clean_text(p.get("title") or "Unknown"), 65)
                year = p.get("year") or "n.d."
                journal = self._truncate(self._clean_text(p.get("journal") or "—"), 25)
                url = p.get("url", "")
                title_cell = f"[{title}]({url})" if url else title
                sections.append(f"| {i} | {title_cell} | {year} | {journal} |")
            sections.append("")
        else:
            sections.append("*No papers retrieved.*\n")

        sections += ["---", ""]

        # ── Section 3: Hypotheses ────────────────────────────────────────────
        sections += [
            "## 3. Generated Research Hypotheses",
            "",
            f"The multi-agent system autonomously generated, critiqued, evolved "
            f"and ranked **{len(top_ideas)} top research hypotheses**.",
            "",
        ]

        for i, idea in enumerate(top_ideas, 1):
            details = (
                idea.get("refined_method_details")
                or idea.get("method_details")
                or {}
            )
            title = self._clean_text(details.get("title") or idea.get("text", "Untitled"))
            score = idea.get("score", 0.0)
            description = self._clean_text(details.get("description") or idea.get("rationale", "—"))
            method = self._clean_text(details.get("method", ""))
            statement = self._clean_text(details.get("statement", ""))

            sections += [
                f"### Hypothesis {i}: {title}",
                "",
                f"**Score**: `{score:.2f} / 10`",
                "",
                "**Description:**",
                "",
                description,
                "",
            ]

            if statement:
                sections += [
                    "**Problem Statement:**",
                    "",
                    statement,
                    "",
                ]

            if method:
                sections += [
                    "**Proposed Methodology:**",
                    "",
                    method,
                    "",
                ]

            # Experiment Design
            exp = idea.get("experiment_design", {})
            if exp:
                sections += [
                    "**Experiment Design:**",
                    "",
                    "- **Datasets:** " + ", ".join(exp.get("datasets", ["N/A"])),
                    "- **Metrics:** " + ", ".join(exp.get("evaluation_metrics", ["N/A"])),
                    "- **Baselines:** " + ", ".join(exp.get("baselines", ["N/A"])),
                    "- **Key Hyperparameters:** " + ", ".join(exp.get("hyperparameters", ["N/A"])),
                    "",
                    "**Implementation Protocol:**",
                    exp.get("implementation_protocol", "N/A"),
                    "",
                ]

            # Supporting refs attached to this idea
            refs = idea.get("references", [])
            if refs:
                sections.append("**Supporting Literature:**")
                for ref in refs[:4]:
                    ref_title = self._clean_text(ref.get("title", ""))
                    ref_url = ref.get("url", "")
                    if ref_title:
                        link = f"[{ref_title}]({ref_url})" if ref_url else ref_title
                        sections.append(f"- {link}")
                sections.append("")

            # Criteria scores if available
            scores_dict = idea.get("scores", {})
            if scores_dict:
                sections.append("**Evaluation Scores:**")
                sections.append("")
                sections.append("| Criterion | Score |")
                sections.append("|-----------|-------|")
                for criterion, val in scores_dict.items():
                    sections.append(f"| {criterion} | {val:.2f} |")
                sections.append("")

            sections += ["---", ""]

        # ── Section 4: Methodology ───────────────────────────────────────────
        sections += [
            "## 4. System Methodology",
            "",
            "This report was produced by an autonomous multi-agent research pipeline:",
            "",
            "```",
            "Phase 1 ─ Deep Research Agent",
            "         └─ Searches arXiv + Semantic Scholar",
            "         └─ LLM synthesizes domain background",
            "",
            "Phase 2 ─ Multi-Agent Hypothesis Generation",
            "         └─ Generation Agent    : creates initial hypotheses",
            "         └─ Reflection Agent   : critiques novelty & feasibility",
            "         └─ Scholar Agent      : retrieves supporting evidence",
            "         └─ Evolution Agent    : refines hypotheses via critique",
            "         └─ Ranking Agent      : scores and selects top N",
            "         └─ Method Dev. Agent  : builds detailed methodology",
            "         └─ Refinement Agent   : polishes final proposals",
            "         └─ Experiment Agent   : designs reproducible protocols",
            "",
            "Phase 3 ─ Report Generator",
            "         └─ Assembles everything into this report",
            "```",
            "",
            "---",
            "",
            f"*Generated by **IdeaGen** on {now}*",
        ]

        return "\n".join(sections)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        return text if len(text) <= max_len else text[: max_len - 1] + "…"

    def save(self, content: str, path: str) -> None:
        """Write the report Markdown to a file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)