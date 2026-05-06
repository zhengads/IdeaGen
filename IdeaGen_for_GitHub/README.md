# IdeaGen: Autonomous Research Agent System

IdeaGen is a Multi-Agent System (MAS) designed to automate the process of scientific literature review, hypothesis generation, and research report creation. It features a fully decoupled visual rendering module that transforms raw agent logs into professional, academic-grade PDF reports.

## Features

*   **Multi-Agent Orchestration**: Specialized agents for deep research, hypothesis generation, critique, and experimental design.
*   **Literature Retrieval**: Integrated search capabilities to pull context from academic databases (e.g., arXiv, Semantic Scholar).
*   **Full-Stack Visualization**: 
    *   Generates exquisite, academic-style PDF reports.
    *   Leverages `WeasyPrint` for advanced CSS-based page layouts (automatic pagination, headers/footers, and complex component rendering).
    *   Integrates `latex2mathml` for high-quality, vector-based rendering of mathematical formulas.
*   **Visual Decoupling**: Separates messy, high-density backend reasoning logs from the final, clean research output.

## Installation

1.  Clone this repository.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The system requires API keys for the language models and search tools to function. 

1.  Create a `.env` file in the root directory.
2.  Add your necessary API keys (do NOT commit this file to version control):
    ```env
    OPENAI_API_KEY=your_openai_api_key_here
    # Add other keys as required by your configuration (e.g., ANTHROPIC_API_KEY, S2_API_KEY)
    ```

## Usage

To start the autonomous research process, run the main entry point script:

```bash
python run_ideas.py
```

The system will execute the configured multi-agent workflow and automatically generate the final PDF report in the designated output directory.

## License

[MIT License](LICENSE)
