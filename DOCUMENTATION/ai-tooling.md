AI Tooling Report for Kolrose Limited RAG System
This document details the AI-powered tools used during the development of this project, evaluating their effectiveness as required by the assignment rubric. The goal was to leverage these tools to accelerate development, improve code quality, and solve complex problems.

1. Code Generation & Conversational AI
1.1. Claude.ai / ChatGPT / Gemini
These were the primary tools used for all stages of the project.

What Worked Well:

Project Scaffolding: Prompts like "Create a production-ready FastAPI and Streamlit app structure for a RAG system" generated the initial app/, frontend/, and backend/ directories, configurations, and boilerplate code, saving hours of manual setup.

Writing Documentation: The detailed README.md and this design-and-evaluation.md were drafted using AI, which structured the raw technical notes into clear, professional documentation for the rubric.

Debugging: When faced with a Segmentation fault from PyTorch on Windows, an LLM correctly diagnosed a likely binary incompatibility and provided the exact pip install command for the CPU-only version of PyTorch, an instant fix after other methods got stuck.

Writing Evaluation Scripts: The logic for the evaluation.py module, which requires decomposing answers into claims and comparing them against source documents, was complex. An AI provided the GroundednessEvaluator class, including claim decomposition and confidence scoring, which only needed minor tuning.

What Didn't Work as Well:

Outdated Imports: The generated code often used import paths from LangChain v0.1, causing ModuleNotFoundError at runtime. All import statements (e.g., langchain.embeddings to langchain_community.embeddings) had to be manually corrected, showing the need for constant developer oversight.

Dependency Conflicts: It would confidently list a full requirements.txt without considering version conflicts. A suggestion to install numPy 2.1 broke compatibility with pandas 2.2.0. The developer needed to research and pin a compatible version (numPy 2.0.2) to resolve the binary conflict.

2. AI-Enhanced IDE (GitHub Copilot / Cursor)
What Worked Well:

Autocomplete for Boilerplate: Copilot was excellent at auto-completing repetitive code like class methods (e.g., all the __init__ and to_dict for dataclasses) and standard functions, saving significant typing time.

Docstring Generation: It automatically generated structured docstrings for functions, which was very helpful when documenting the API endpoints in main.py.

What Didn't Work as Well:

Incorrect Business Logic: When writing the TopicClassifier, it would often suggest catch-all keywords that were incorrect for company policies. For example, it once suggested "banana" as an off-topic indicator, which shows it needs constant supervision for domain-specific logic.

3. Specialized AI (RAG & Search Assistants)
Search Assistants (Gemini/Claude with internet access):

What Worked Well: These were extremely helpful for researching specific documentation and configuration details. They could quickly fetch the exact setup for a ChromaDB persistent client or the most up-to-date OpenRouter free model names, which was more efficient than manually browsing through documentation pages.

4. Summary of Impact
Acceleration: AI tools were critical in meeting the project deadline. They compressed the time for boilerplate tasks, documentation, and initial research.

Quality: The generated code provided a solid, well-structured foundation, but the final quality and correctness of the application were entirely dependent on the developer’s oversight, debugging skills, and subject-matter expertise.

The key lesson learned is that these AI tools act as very capable junior developers but require an experienced engineer to architect the system, verify every suggestion, and fix the inevitable errors. They are a powerful toolset but cannot replace the developer's critical thinking.