"""
Premium PDF Report Exporter for IdeaGen (WeasyPrint Version)

Converts the structured research report into a high-end, professional academic-style PDF.
Uses WeasyPrint for robust HTML/CSS layout and latex2mathml for vector-based 
mathematical formulas.
"""

import markdown
import latex2mathml.converter
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration
import re
import os
import datetime
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

def markdown_to_html_with_math(text: str) -> str:
    """
    Converts markdown text to HTML, ensuring LaTeX formulas are converted to MathML
    for high-quality rendering in WeasyPrint.
    """
    if not text:
        return ""
    
    # Pre-process math blocks and inline math
    # 1. Handle block math: $$ formula $$
    text = re.sub(r'\$\$(.*?)\$\$', 
                  lambda m: f'<div class="math-block">{latex2mathml.converter.convert(m.group(1))}</div>', 
                  text, flags=re.DOTALL)
    
    # 2. Handle inline math: $ formula $
    text = re.sub(r'\$(.*?)\$', 
                  lambda m: f'<span class="math-inline">{latex2mathml.converter.convert(m.group(1))}</span>', 
                  text)
    
    # 3. Basic Markdown cleanup for common LLM artifacts
    # Replace single newlines with spaces within paragraphs, but preserve double newlines
    # (Simplified approach: markdown library handles most of this)
    
    # 4. Use markdown library to generate HTML structure
    html = markdown.markdown(text, extensions=['extra', 'codehilite', 'tables', 'toc'])
    return html

def generate_report_pdf(
    topic: str,
    domain: str,
    deep_research: Dict[str, Any],
    top_ideas: List[Dict[str, Any]],
    output_path: str,
) -> str:
    """
    Generates a premium PDF report using WeasyPrint.
    """
    logger.info(f"Generating premium PDF report via WeasyPrint: {output_path}")

    # --- Data Preparation ---
    
    # Literatue References Section
    papers = deep_research.get("papers", [])[:10]
    papers_rows = ""
    for i, p in enumerate(papers, 1):
        title = p.get("title", "Unknown Title")
        year = p.get("year", "N/A")
        journal = p.get("journal", p.get("venue", "N/A"))
        papers_rows += f"""
        <tr>
            <td class="idx">{i}</td>
            <td class="title">{title}</td>
            <td class="year">{year}</td>
            <td class="source">{journal}</td>
        </tr>
        """

    # Hypotheses Section
    hypotheses_html = ""
    for i, idea in enumerate(top_ideas, 1):
        # Extract refined details if available, otherwise original
        details = idea.get("refined_method_details", idea.get("method_details", {}))
        if not details:
            details = {"title": idea.get("text", "Untitled"), "description": idea.get("rationale", "")}
            
        title = details.get("title", "Untitled Hypothesis")
        description = details.get("description", "")
        statement = details.get("statement", "")
        method = details.get("method", "")
        
        # Scoring metrics
        scores = idea.get("scores", {})
        total_score = round(idea.get("score", 0), 2)
        
        score_html = ""
        for metric, val in scores.items():
            percentage = val * 10 if val <= 10 else 100
            score_html += f"""
            <div class="score-row">
                <span class="metric-name">{metric.capitalize()}</span>
                <div class="progress-bg"><div class="progress-fill" style="width: {percentage}%"></div></div>
                <span class="metric-val">{val}/10</span>
            </div>
            """
            
        # Experiment Design (if exists)
        exp = idea.get("experiment_design", {})
        exp_html = ""
        if exp:
            exp_html = f"""
            <div class="experiment-box">
                <h4>Proposed Experimental Setup</h4>
                <p><strong>Benchmarks:</strong> {", ".join(exp.get("datasets", ["N/A"]))}</p>
                <p><strong>Metrics:</strong> {", ".join(exp.get("evaluation_metrics", ["N/A"]))}</p>
                <p><strong>Protocol:</strong> {exp.get("implementation_protocol", "N/A")}</p>
            </div>
            """

        hypotheses_html += f"""
        <div class="idea-container">
            <div class="idea-header">
                <span class="idea-num">Hypothesis {i}</span>
                <span class="total-score">Overall Score: {total_score}</span>
            </div>
            <h3>{title}</h3>
            
            <div class="scores-grid">
                {score_html}
            </div>

            <div class="section-content">
                <strong>Rationale & Motivation</strong>
                {markdown_to_html_with_math(description)}
            </div>
            
            {"<div class='section-content'><strong>Problem Statement</strong>" + markdown_to_html_with_math(statement) + "</div>" if statement else ""}
            
            {"<div class='section-content method-highlight'><strong>Technical Methodology</strong>" + markdown_to_html_with_math(method) + "</div>" if method else ""}
            
            {exp_html}
        </div>
        """

    # --- HTML Styling (CSS) ---
    css_styles = """
    @page {
        size: A4;
        margin: 2.5cm 2cm;
        @bottom-right {
            content: "Page " counter(page) " of " counter(pages);
            font-size: 8pt;
            color: #777;
        }
        @top-left {
            content: "IdeaGen Research Report";
            font-size: 8pt;
            color: #aaa;
            font-style: italic;
        }
    }

    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.5;
        color: #2c3e50;
        background-color: white;
        margin: 0;
        font-size: 10.5pt;
    }

    /* Typography */
    h1 { font-size: 32pt; margin-top: 0; color: #1a2a6c; letter-spacing: -1px; line-height: 1.1; }
    h2 { font-size: 18pt; color: #1a2a6c; border-bottom: 2px solid #eee; padding-bottom: 5px; margin-top: 40px; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px; }
    h3 { font-size: 15pt; color: #34495e; margin: 10px 0; line-height: 1.3; }
    h4 { font-size: 11pt; color: #2c3e50; margin-bottom: 10px; margin-top: 0; text-transform: uppercase; font-weight: 700; }

    p { margin-bottom: 12px; text-align: justify; }
    
    /* Layout Components */
    .title-area { 
        padding: 40px 0; 
        margin-bottom: 30px; 
        background: linear-gradient(to right, #ffffff, #f8f9fa);
        border-bottom: 5px solid #1a2a6c;
    }
    
    .meta-infobar { 
        display: flex; 
        justify-content: space-between; 
        font-size: 9pt; 
        color: #666; 
        background: #f1f3f4; 
        padding: 8px 15px; 
        border-radius: 4px;
        margin-bottom: 25px;
    }

    /* Table Styling */
    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
    th { background-color: #f8f9fa; color: #1a2a6c; text-align: left; padding: 10px; font-size: 9pt; border-bottom: 2px solid #dee2e6; }
    td { padding: 10px; border-bottom: 1px solid #eee; font-size: 9pt; vertical-align: top; }
    .idx { width: 30px; color: #999; }
    .year { width: 40px; text-align: center; }
    .source { width: 120px; font-style: italic; color: #555; }

    /* Idea Card Styling */
    .idea-container { 
        margin-bottom: 45px; 
        padding: 25px; 
        border: 1px solid #eaebec; 
        border-radius: 8px; 
        background-color: #fdfdfd;
        break-inside: avoid;
        box-shadow: 0 4px 12px rgba(0,0,0,0.02);
    }
    
    .idea-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; border-bottom: 1px dashed #ddd; padding-bottom: 8px; }
    .idea-num { font-weight: bold; color: #e67e22; font-size: 10pt; text-transform: uppercase; }
    .total-score { font-weight: bold; background: #1a2a6c; color: white; padding: 3px 10px; border-radius: 20px; font-size: 9pt; }

    .section-content { margin: 15px 0; }
    .section-content strong { display: block; margin-bottom: 8px; color: #1a2a6c; font-size: 9.5pt; text-transform: uppercase; letter-spacing: 0.5px; }

    .method-highlight { 
        background: #f0f7ff; 
        padding: 15px; 
        border-left: 4px solid #3498db; 
        border-radius: 0 4px 4px 0;
    }

    .experiment-box {
        margin-top: 20px;
        background: #fdfae6;
        padding: 15px;
        border: 1px solid #f3e5ab;
        border-radius: 6px;
        font-size: 9.5pt;
    }

    /* Scores Grid */
    .scores-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 25px; margin: 15px 0; }
    .score-row { display: flex; align-items: center; }
    .metric-name { width: 100px; font-size: 8.5pt; color: #7f8c8d; }
    .progress-bg { flex-grow: 1; height: 6px; background: #ecf0f1; border-radius: 3px; margin: 0 10px; overflow: hidden; }
    .progress-fill { height: 100%; background: #27ae60; border-radius: 3px; }
    .metric-val { width: 40px; font-size: 8.5pt; font-weight: bold; color: #2ecc71; text-align: right; }

    /* Math Styling */
    math { font-size: 110%; color: #c0392b; }
    .math-block { margin: 20px 0; text-align: center; overflow-x: auto; background: #fff; padding: 10px; }
    .math-inline { color: #c0392b; }

    /* Footer Meta */
    .report-footer {
        margin-top: 60px;
        text-align: center;
        font-size: 8pt;
        color: #999;
        border-top: 1px solid #eee;
        padding-top: 20px;
    }
    """

    # --- HTML Assembly ---
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <style>{css_styles}</style>
    </head>
    <body>
        <div class="title-area">
            <h1>Research Report:<br>{topic}</h1>
        </div>

        <div class="meta-infobar">
            <span><strong>Domain:</strong> {domain}</span>
            <span><strong>Analysis Date:</strong> {now}</span>
            <span><strong>References:</strong> {len(papers)}</span>
        </div>

        <h2>I. Introduction & Literature Synthesis</h2>
        <div class="synthesis-content">
            {markdown_to_html_with_math(deep_research.get("synthesis", "N/A"))}
        </div>

        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Key Reference Title</th>
                    <th>Year</th>
                    <th>Scientific Journal / Source</th>
                </tr>
            </thead>
            <tbody>
                {papers_rows}
            </tbody>
        </table>

        <div style="page-break-after: always;"></div>

        <h2>II. Proposed Research Hypotheses</h2>
        <p>The following {len(top_ideas)} hypotheses were generated and refined based on the literature synthesis above, 
        prioritizing novelty, feasibility, and technical depth.</p>
        
        {hypotheses_html}

        <div class="report-footer">
            Generated autonomously by IdeaGen Research Pipeline • Multi-Agent System Workflow<br>
            © {datetime.datetime.now().year} InternAgent Project
        </div>
    </body>
    </html>
    """

    # --- PDF Generation ---
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        font_config = FontConfiguration()
        HTML(string=html_content).write_pdf(
            output_path, 
            font_config=font_config
        )
        logger.info("PDF generated successfully via WeasyPrint.")
    except Exception as e:
        logger.error(f"WeasyPrint PDF generation failed: {str(e)}")
        raise

    return output_path