"""
Deep Research Agent for IdeaGen

Performs autonomous literature review across arXiv and Semantic Scholar,
then synthesizes findings into a structured research background using LLM.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import litellm
from litellm import acompletion

from internagent.mas.tools.literature_search import LiteratureSearch

# Disable remote cost map fetching up-front to avoid SSL handshake timeout warnings
litellm.disable_remote_host_map = True

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

SYNTHESIS_PROMPT = """You are a senior research scientist conducting a deep literature review to establish a foundation for novel hypothesis generation.

Research Topic: {topic}
Domain: {domain}
{background_section}

Based on the highly relevant papers retrieved from academic databases:

{papers_text}

Please provide a highly structured and comprehensive research synthesis (500-800 words) that MUST cover the following sections:

1. **Current Research Landscape & SOTA**: Categorize existing methods into distinct schools of thought (clutering). What are the main-stream baseline approaches?
2. **Core Complexity & Technical Pain Points**: Analyze why current methods (e.g., vanilla attention) fail in this specific domain. Detail the complexity bottlenecks (e.g., O(N^2)).
3. **Existing Method Limitations**: Why are current SOTA solutions insufficient? Mention specific trade-offs (e.g., accuracy vs. efficiency).
4. **Significant Research Value & Gaps**: Why is exploring this topic meaningful right now? Which "unoccupied" niche or technical route needs immediate attention?

Write this synthesis professionally, using LaTeX for any mathematical notations or complexity classes. This report serves as the 'Environment Background' for the IdeaGen agents.
"""


# ── Main class ────────────────────────────────────────────────────────────────

class DeepResearcher:
    """
    Autonomous deep research agent.

    Searches arXiv and Semantic Scholar for relevant papers,
    then uses an LLM to synthesize the findings into a structured
    research background that feeds into hypothesis generation.
    """

    # Number of papers to retrieve per query per source
    PAPERS_PER_QUERY = 8
    MAX_TOTAL_PAPERS = 30
    MAX_PAPERS_FOR_LLM = 20   # feed at most this many abstracts to LLM
    ABSTRACT_PREVIEW_LEN = 600

    def __init__(
        self,
        topic: str,
        domain: str,
        config: Dict[str, Any],
        background: str = "",
    ):
        self.topic = topic
        self.domain = domain
        self.config = config
        self.background = background
        self._model_cfg = self._resolve_model(config)

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(self) -> Dict[str, Any]:
        """
        Execute deep research pipeline.

        Returns:
            dict with keys: topic, domain, paper_count, papers, synthesis
        """
        logger.info(f"[DeepResearch] Starting research on: '{self.topic}'")

        papers = await self._search_papers()
        logger.info(f"[DeepResearch] Collected {len(papers)} unique papers")

        synthesis = await self._synthesize(papers)

        return {
            "topic": self.topic,
            "domain": self.domain,
            "paper_count": len(papers),
            "papers": [self._paper_to_dict(p) for p in papers],
            "synthesis": synthesis,
        }

    # ── Private: Search ───────────────────────────────────────────────────────

    async def _generate_search_queries(self) -> List[str]:
        """Generate precise, domain-specific search queries using LLM."""
        prompt = f"""You are an expert scientific researcher. Generate 4 highly precise search queries for academic databases (like arXiv) to find foundational papers for the following topic.

Topic: {self.topic}
Domain: {self.domain}

Guidelines:
1. Use specific technical terminology from the {self.domain} domain.
2. Do NOT use generic terms like "AI for chemistry". Instead, use terms like `"machine learning" AND "molecular dynamics"`.
3. Use AND/OR boolean operators where appropriate (e.g. `"deep learning" AND "drug discovery"`).
4. Output EXACTLY 4 queries, one per line. Do not include numbering, bullet points, or any other text.
"""
        try:
            kwargs: Dict[str, Any] = {
                "model": self._model_cfg["model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
                "max_tokens": 150,
                "timeout": 60,
            }
            if self._model_cfg.get("api_base"):
                kwargs["api_base"] = self._model_cfg["api_base"]

            response = await acompletion(**kwargs)
            content = response.choices[0].message.content.strip()
            queries = [q.strip() for q in content.split('\n') if q.strip()]
            
            # Ensure we have some queries
            if not queries:
                raise ValueError("No queries generated")
                
            logger.info(f"[DeepResearch] Generated {len(queries)} specific search queries:\n" + "\n".join(queries))
            # Limit to 4 queries to manage API calls
            return queries[:4]
            
        except Exception as e:
            logger.warning(f"[DeepResearch] Failed to generate queries: {e}. Falling back to default queries.")
            return [
                self.topic,
                f'"{self.topic}" AND "{self.domain}"'
            ]

    async def _search_papers(self) -> list:
        """Search multiple academic databases with parallelized queries and semantic filtering."""
        searcher = LiteratureSearch(email="research@ideagen.ai")

        queries = await self._generate_search_queries()

        seen: set = set()
        raw_papers: list = []

        def _dedup_add(results):
            """Add results to raw list, deduplicating by title."""
            for p in results:
                key = getattr(p, "title", "").lower().strip()
                if key and key not in seen:
                    seen.add(key)
                    raw_papers.append(p)

        # ── Batch 2: arXiv (all queries concurrently) ─────────────────────
        logger.info("[DeepResearch] Searching arXiv (all queries concurrently)...")

        async def _search_arxiv(q):
            try:
                return await searcher.search_arxiv(q, max_results=self.PAPERS_PER_QUERY)
            except Exception as e:
                logger.warning(f"  -> arXiv error for '{q}': {e}")
                return []

        arxiv_results = await asyncio.gather(*[_search_arxiv(q) for q in queries])
        for result_list in arxiv_results:
            _dedup_add(result_list)

        logger.info(f"[DeepResearch] Total unique papers before filtering: {len(raw_papers)}")

        # ── Semantic Filtering ──
        filtered_papers = await self._filter_irrelevant_papers(raw_papers)
        logger.info(f"[DeepResearch] Papers after semantic filtering: {len(filtered_papers)}")

        return filtered_papers[:self.MAX_TOTAL_PAPERS]

    async def _filter_irrelevant_papers(self, papers: list) -> list:
        """Filter out papers that are not relevant to the target domain/topic using LLM."""
        if not papers:
            return []
            
        # Batch processing to save tokens/time
        BATCH_SIZE = 10
        filtered = []
        
        logger.info(f"[DeepResearch] Semantic filtering {len(papers)} papers...")
        
        for i in range(0, len(papers), BATCH_SIZE):
            batch = papers[i:i+BATCH_SIZE]
            paper_meta = "\n".join([f"{idx+1}. Title: {p.title} | Abstract: {p.abstract[:self.ABSTRACT_PREVIEW_LEN]}..." for idx, p in enumerate(batch) if hasattr(p, 'title')])
            
            prompt = f"""You are a strict scientific literature filter. Decide the relevance of each paper to the research topic.

### Target Context
Topic: {self.topic}
Domain: {self.domain}

    1. **Domain Consistency**: The paper MUST be within or directly applicable to the target domain ({self.domain}). 
    2. **Context Alignment**: Reject papers that apply similar methods and techniques to application areas fundamentally different from the target domain ({self.domain}). For example, if the domain is Chemistry, papers about AI ethics, astronomy, or NLP are completely irrelevant.
    3. **Topic Relevance**: The paper should address the specific topic ({self.topic}) or provide essential methodological baselines that can be directly mapped to the target goal.

### List of papers:
{paper_meta}

Output ONLY a comma-separated list of scores for each of the {len(batch)} papers in order, using this scale:
0 = Irrelevant (wrong domain, e.g. ethics/astronomy/security)
1 = Possibly relevant (right domain, related methods)
2 = Highly relevant (directly addresses the topic)

Example response for 3 papers: 2, 0, 1
No other text, explanations, or formatting.
"""
            try:
                kwargs: Dict[str, Any] = {
                    "model": self._model_cfg["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 100,
                    "timeout": 60,
                }
                if self._model_cfg.get("api_base"):
                    kwargs["api_base"] = self._model_cfg["api_base"]

                response = await acompletion(**kwargs)
                content = response.choices[0].message.content.strip()
                logger.info(f"[DeepResearch] Filter response: {content}")
                
                # Robust parsing: find all 0, 1, 2
                import re
                scores = re.findall(r'[012]', content)
                
                if len(scores) < len(batch):
                    logger.warning(f"[DeepResearch] Filter returned fewer scores ({len(scores)}) than batch size ({len(batch)}). Content: {content}")
                
                for idx, score in enumerate(scores):
                    if idx < len(batch) and int(score) >= 1:
                        filtered.append(batch[idx])
            except Exception as e:
                logger.warning(f"  -> Semantic filtering error for batch: {e}")
                # Don't fallback, skip the batch if it fails to avoid adding noise.
                pass
        
        # We removed the fallback that keeps all papers if none pass. 
        # If 0 papers pass, the queries were likely poor, and the LLM synthesis will handle the empty result.
                
        return filtered

    # ── Private: Synthesis ────────────────────────────────────────────────────

    async def _synthesize(self, papers: list) -> str:
        """Use LLM to synthesize paper findings into a research background."""
        if not papers:
            fallback = (
                f"Research area: {self.topic} (domain: {self.domain}). "
                "No papers were retrieved during the search. "
                "Please broaden the topic or check network connectivity."
            )
            logger.warning("[DeepResearch] No papers found – returning fallback synthesis.")
            return fallback

        papers_text = self._format_papers_for_prompt(papers[: self.MAX_PAPERS_FOR_LLM])
        background_section = (
            f"\nAdditional Background Provided:\n{self.background}\n"
            if self.background
            else ""
        )

        prompt = SYNTHESIS_PROMPT.format(
            topic=self.topic,
            domain=self.domain,
            background_section=background_section,
            papers_text=papers_text,
        )

        try:
            kwargs: Dict[str, Any] = {
                "model": self._model_cfg["model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 2048,
                "timeout": 300,
            }
            if self._model_cfg.get("api_base"):
                kwargs["api_base"] = self._model_cfg["api_base"]

            response = await acompletion(**kwargs)
            synthesis = response.choices[0].message.content
            logger.info("[DeepResearch] LLM synthesis completed successfully")
            return synthesis

        except Exception as exc:
            logger.error(f"[DeepResearch] LLM synthesis failed: {exc}")
            return (
                f"Research topic: {self.topic}. Domain: {self.domain}. "
                f"{len(papers)} papers were found but LLM synthesis failed: {exc}"
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _format_papers_for_prompt(self, papers: list) -> str:
        lines = []
        for i, p in enumerate(papers, 1):
            d = self._paper_to_dict(p)
            title = d.get("title") or "Untitled"
            year = d.get("year") or "n.d."
            journal = d.get("journal") or "—"
            abstract = (d.get("abstract") or "")[:self.ABSTRACT_PREVIEW_LEN]
            lines.append(
                f"{i}. [{year}] {title}  ({journal})\n"
                f"   Abstract: {abstract}...\n"
            )
        return "\n".join(lines)

    @staticmethod
    def _paper_to_dict(paper) -> Dict[str, Any]:
        if hasattr(paper, "to_dict"):
            return paper.to_dict()
        return paper if isinstance(paper, dict) else {}

    @staticmethod
    def _resolve_model(config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model name and api_base from config."""
        models = config.get("models", {})
        provider = models.get("default_provider", "openai")
        provider_conf = models.get(provider, {})
        model_name = provider_conf.get("model_name", "gpt-4o-mini")
        api_base = provider_conf.get("api_base") or os.getenv("OPENAI_API_BASE_URL")
        
        # litellm 需要 openai/ 前缀来识别第三方兼容接口，但官方 openai sdk 不需要
        # 此时我们在 deep_research 内部作转换拦截
        litellm_model = model_name
        if provider == "openai" and not model_name.startswith("openai/"):
            litellm_model = f"openai/{model_name}"
            
        return {"model": litellm_model, "api_base": api_base}
