"""
Experiment Design Agent for IdeaGen

Generates detailed experimental protocols for research hypotheses, including
evaluation metrics, dataset selection, and baseline comparison plans.
"""

import logging
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentExecutionError

logger = logging.getLogger(__name__)

class ExperimentDesignAgent(BaseAgent):
    """
    Agent that generates reproducible experimental designs for research hypotheses.
    """

    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)

    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate experimental design for a given hypothesis.

        Args:
            context: Dictionary containing:
                - goal: Research goal
                - hypothesis: Hypothesis to design experiment for
                - synthesis: Global literature synthesis (if available)

        Returns:
            Dictionary containing experimental protocol.
        """
        goal = context.get("goal", {})
        hypothesis = context.get("hypothesis", {})
        synthesis = context.get("synthesis", "")

        prompt = self._build_experiment_prompt(goal, hypothesis, synthesis)
        system_prompt = self._build_system_prompt()

        output_schema = {
            "type": "object",
            "properties": {
                "experiment_design": {
                    "type": "object",
                    "properties": {
                        "hyperparameters": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Key hyperparameters settings (e.g. learning rate, batch size, architectural dims)"
                        },
                        "datasets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Benchmark datasets to use for validation"
                        },
                        "evaluation_metrics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Primary and secondary metrics (e.g. Perplexity, Throughput, Memory Usage)"
                        },
                        "baselines": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Standard models/approaches to compare against"
                        },
                        "ablation_plan": {
                            "type": "string",
                            "description": "Description of planned ablation studies"
                        },
                        "implementation_protocol": {
                            "type": "string",
                            "description": "Step-by-step protocol for reproducing the experiment"
                        }
                    },
                    "required": ["hyperparameters", "datasets", "evaluation_metrics", "baselines", "implementation_protocol"]
                }
            },
            "required": ["experiment_design"]
        }

        try:
            response = await self._call_model(
                prompt=prompt,
                system_prompt=system_prompt,
                schema=output_schema
            )
            return response
        except Exception as e:
            logger.error(f"Experiment design generation failed: {str(e)}")
            raise AgentExecutionError(f"Failed to generate experiment design: {str(e)}")

    def _build_experiment_prompt(self, goal, hypothesis, synthesis) -> str:
        return f"""You are a scientific experiment design specialist. Your goal is to create a rigorous, reproducible experimental protocol for the following hypothesis.

Research Goal: {goal.get('description', '')}
Current Hypothesis: {hypothesis.get('text', '')}
Technical Rationale: {hypothesis.get('rationale', '')}

Background Context:
{synthesis}

Please design an experiment that can validly test this hypothesis. Focus on:
1. **Measurability**: How exactly will we measure the efficiency gain or performance improvement?
2. **Fair Comparison**: Which baselines are necessary to prove the novelty?
3. **Reproducibility**: Provide specific hyperparameters and dataset choices (e.g., SlimPajama, LongBench).
4. **Structural Validation**: How will ablation studies confirm that the specific innovation is responsible for the result?
"""

    def _build_system_prompt(self) -> str:
        return """You are a senior experimental physicist and machine learning researcher.
Your task is to design watertight, reproducible experimental protocols.
Strictly adhere to the following principles:
- **Baseline Integrity**: Always compare against the most relevant SOTA baselines.
- **Metric Rigor**: Use standard metrics (e.g., for Long-context: RULER score, Passkey retrieval, complexity scaling).
- **Hypothesis Isolation**: Ensure the experiment isolates the effect of the proposed architectural change.
- **Precision**: Provide concrete numbers/ranges for hyperparameters where possible.
"""

    @classmethod
    def from_config(cls, config: Dict[str, Any], model: Any) -> 'ExperimentDesignAgent':
        return cls(model, config)
