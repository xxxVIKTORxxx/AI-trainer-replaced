# system_blueprint.py
import os
import json
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

# Ensure you install these dependencies:
# pip install crewai langchain-google-genai pydantic

try:
    from crewai import Agent, Crew, Process, Task
    from crewai.tools import tool
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    print("Dependencies missing! Run: pip install crewai langchain-google-genai")
    # Mocking for standalone script integrity if executed without environment
    class Agent: pass
    class Task: pass
    class Crew: pass
    class Process: pass
    def tool(func): return func

# ----------------------------------------------------
# 1. Models & Structured Output Schemas
# ----------------------------------------------------
class GrammarReport(BaseModel):
    language_detected: str = Field(..., description="The ISO language code or name detected.")
    errors_found: list[str] = Field(..., description="List of concrete typos, syntax issues, or clumsy phrases.")
    suggestions: list[str] = Field(..., description="Actionable edit suggestions corresponding to the errors.")
    quality_score: float = Field(..., description="Grammatical correctness score from 0.0 (poor) to 1.0 (flawless).")

class EthicsReport(BaseModel):
    cultural_context: str = Field(..., description="Identified cultural paradigm or regional lens applied.")
    ethical_violations: list[str] = Field(..., description="List of explicit norm violations, offensive subtext, or biases.")
    safety_score: float = Field(..., description="Ethical alignment score from 0.0 (toxic/unsafe) to 1.0 (fully aligned).")

class LegalReport(BaseModel):
    jurisdiction: str = Field(..., description="The geographical or political legal jurisdiction checked.")
    liability_risks: list[str] = Field(..., description="Potential legal vulnerabilities, copyright red flags, or regulatory breaches.")
    is_compliant: bool = Field(..., description="True if no major legal threats are found.")

# ----------------------------------------------------
# 2. Dynamic, Universal AI-Trainer Tools
# ----------------------------------------------------
@tool("Universal Rule Scraper")
def universal_rule_scraper(text: str) -> str:
    """Scrapes or retrieves explicit grammar rules and cultural norms tailored to the language present in the text."""
    # In production, this can connect to localized wikis, dictionaries, or multi-lingual vector stores.
    # For testing, it acts as an abstraction layer reducing token burden on small/free models.
    return (
        "Linguistic Baseline: Verify standard verb conjugations, gender agreements, and sentence structures.\n"
        "Ethical Baseline: Respect localized professional etiquette, refrain from hate speech, avoid structural biases, "
        "and preserve contextually appropriate safety boundaries."
    )

# ----------------------------------------------------
# 3. System Configuration Factory
# ----------------------------------------------------
def build_ai_trainer_system(
    api_key: str, 
    model_name: str = "gemini-1.5-flash", 
    enable_lawyer: bool = False,
    legal_jurisdiction: str = "Global / General Compliance"
) -> Crew:
    """
    Constructs an ad-hoc, modular cybernetic AI-Trainer crew.
    Highly adaptable for local developers by toggling configuration parameters.
    """
    
    # Initialize the Free-Tier LLM (Gemini 1.5 Flash is ideal for fast, cost-effective agent tasks)
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2 # Lower temperature guarantees structured, deterministic critiques
    )
    
    # --- Agent Definitions ---
    
    grammar_critic = Agent(
        role="Universal Linguistic Quality Coach",
        goal="Identify typos, grammatical anomalies, syntax friction, and language nuances regardless of the language used.",
        backstory="An elite, hyper-polyglot editor who evaluates text purely through structural rules, refusing to summarize or conversationalize.",
        verbose=True,
        llm=llm,
        tools=[universal_rule_scraper]
    )
    
    ethicist_agent = Agent(
        role="Localized Cultural and Ethics Evaluator",
        goal="Audit text for safety, bias, offensive subtext, and alignment with regional or universal ethical paradigms.",
        backstory="An anthropologist and safety researcher specializing in checking alignment across shifting societal frameworks and localized nuances.",
        verbose=True,
        llm=llm
    )
    
    master_editor = Agent(
        role="Master Synthesizer and Dataset Compiler",
        goal="Reconcile all corrections seamlessly, generating a pristine optimized response paired with a metadata scorecard.",
        backstory="A systems manager that fuses separate linguistic and ethical reports back into human-like, balanced prose.",
        verbose=True,
        llm=llm
    )
    
    # --- Task Framework ---
    
    tasks = []
    
    grammar_task = Task(
        description=(
            "Thoroughly analyze the target text: '{target_text}'.\n"
            "Use the Universal Rule Scraper to fetch context if necessary. Identify errors and give precise scores."
        ),
        expected_output="A structured report mapping linguistic bugs and structural suggestions.",
        output_json=GrammarReport,
        agent=grammar_critic
    )
    tasks.append(grammar_task)
    
    ethics_task = Task(
        description=(
            "Evaluate the target text: '{target_text}'.\n"
            "Check for hidden biases, microaggressions, cultural faux pas, or general safety risks."
        ),
        expected_output="A structured report highlighting normative alignments and context.",
        output_json=EthicsReport,
        agent=ethicist_agent
    )
    tasks.append(ethics_task)
    
    # Optional Legal Architecture Layer (Easily toggled on/off by local devs)
    if enable_lawyer:
        lawyer_agent = Agent(
            role="Corporate Legal Risk Assessor",
            goal=f"Scan text for operational liabilities, compliance breaches, and copyright red flags under {legal_jurisdiction} scope.",
            backstory="A strict corporate compliance attorney focusing on risk mitigation, liability avoidance, and regulatory rules.",
            verbose=True,
            llm=llm
        )
        
        legal_task = Task(
            description=(
                f"Verify if the text: '{{target_text}}' exposes the system to legal action or violates regulatory rules within {legal_jurisdiction}."
            ),
            expected_output="A strict legal compliance scorecard highlighting liability risks.",
            output_json=LegalReport,
            agent=lawyer_agent
        )
        tasks.append(legal_task)
        
    # Final Output Fusion Task
    synthesis_task = Task(
        description=(
            "Review the original raw text: '{target_text}'.\n"
            "Incorporate all correction inputs from the previous analysis steps.\n"
            "Generate a final, fully corrected version of the text that fixes grammar and resolves ethical issues "
            "while maintaining the original voice and core intent."
        ),
        expected_output="JSON object containing the pristine corrected text and an array of overall feedback summaries.",
        agent=master_editor
    )
    tasks.append(synthesis_task)
    
    # Combine into a unified cybernetic evaluation group
    training_crew = Crew(
        agents=[grammar_critic, ethicist_agent] + ([lawyer_agent] if enable_lawyer else []) + [master_editor],
        tasks=tasks,
        process=Process.sequential, # Sequential allows downstream editors to look back at upstream raw reports
        verbose=True
    )
    
    return training_crew

# ----------------------------------------------------
# 4. Local Execution / Testing Sandbox
# ----------------------------------------------------
if __name__ == "__main__":
    # To run this, replace with your actual free Google AI Studio API key
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_FREE_GEMINI_API_KEY_HERE")
    
    # A dirty input containing both explicit grammatical errors and highly questionable workplace ethics
    raw_ai_output = (
        "Yesterday my manager thinked that we should fire Bob because he is becoming too old "
        "and old people cannot writing modern fast Python code for our startup."
    )
    
    print(f"--- Launching AI Trainer System ---")
    print(f"Target Draft for Review: \"{raw_ai_output}\"\n")
    
    if GOOGLE_API_KEY == "YOUR_FREE_GEMINI_API_KEY_HERE":
        print("[!] Execution Paused: Please set your GOOGLE_API_KEY to test the live API.")
    else:
        # Build the architecture with the custom Lawyer Agent enabled
        trainer_engine = build_ai_trainer_system(
            api_key=GOOGLE_API_KEY, 
            enable_lawyer=True, 
            legal_jurisdiction="US Employment Law / EEOC"
        )
        
        # Execute the loop
        result = trainer_engine.kickoff(inputs={"target_text": raw_ai_output})
        
        print("\n--- Final Optimized Training Output ---")
        print(result)
