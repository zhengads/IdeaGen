import os.path as osp
import json

from internagent.mas.interface import InternAgentInterface
from internagent.vis import visualize_hypotheses


class IdeaGenerator:
    """Handles idea generation using MAS (Multi-Agent System)"""

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.interface = InternAgentInterface(
            args.config,
            work_dir=args.task_dir,
            task_name=args.task_name
        )
        self.session_id = None
        self.status = None

    async def load_task(self):
        """Load task and create MAS session"""
        self.logger.info(f"Creating research session for: {self.args.task_dir}")

        await self.interface.startup()

        task_desc_path = osp.join(self.args.task_dir, "prompt.json")
        if not osp.exists(task_desc_path):
            raise FileNotFoundError(f"Task description not found: {task_desc_path}")

        with open(task_desc_path, 'r') as f:
            params = json.load(f)

        goal = params.get('task_description')
        domain = params.get('domain')
        background = params.get('background', "")
        constraints = params.get('constraints', [])

        if not goal or not domain:
            raise ValueError("Task description and domain are required")

        self.session_id = await self.interface.create_session(
            goal_description=goal,
            domain=domain,
            background=background,
            ref_code_path=getattr(self.args, 'ref_code_path', None) or "",
            constraints=constraints
        )

        self.logger.info(f"Session created: {self.session_id}")

    async def generate_ideas(self):
        """Run MAS to generate and rank research ideas/hypotheses"""
        if self.session_id is None:
            await self.load_task()

        await self.interface.startup()

        max_iterations = 10

        while self.status != "completed":
            try:
                full_status = await self.interface.get_session_status(self.session_id)
                self.status = full_status['state']
                iterations = full_status['iterations_completed']

                if iterations >= max_iterations:
                    self.logger.warning(f"Maximum iterations ({max_iterations}) reached. Forcing completion to save tokens.")
                    break

                if self.status == "awaiting_feedback":
                    if getattr(self.args, 'offline_feedback', None):
                        with open(self.args.offline_feedback, "r") as f:
                            feedback = json.load(f)
                        await self.interface.add_feedback(self.session_id, feedback)
                        self.logger.info(f"Offline feedback injected: {feedback}")

                elif self.status == "completed":
                    self.logger.info("Idea generation completed")
                    break

                elif self.status == "error":
                    raise RuntimeError("Error in MAS session")

                self.logger.info(f"Running session {self.session_id}, iteration {iterations}")
                self.status = await self.interface.run_session(self.session_id)

            except Exception as e:
                self.logger.error(f"Error in session: {str(e)}")
                raise

        top_ideas = await self.interface.get_top_ideas(self.session_id)
        self.logger.info(f"Generated {len(top_ideas)} top ideas")

        # Save session trajectory
        session_json = osp.join("results", self.args.task_name, f"traj_{self.session_id}.json")
        # Optimization: Disabled Idea_Evolution_Graph.pdf as the methodology section in research report is sufficient.
        # vis_output = osp.join(
        #     "results", self.args.task_name,
        #     "Idea_Evolution_Graph.pdf"
        # )
        # visualize_hypotheses(session_json, vis_output)
        # self.logger.info(f"Visualization saved: {vis_output}")

        return top_ideas, session_json
