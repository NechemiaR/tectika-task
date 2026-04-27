# Agent Architecture and Workflow (No Code)

## 1. Manager / Orchestrator

**Role:** Manages the process, delegates work, and collects outputs. The agent does not generate content itself but coordinates the activity.

* **Initial Input:** Raw research topic (e.g., "The impact of LLMs on backend engineering workflows").

* **Action 1 - Task Analysis:** The Manager passes the main topic to the Planner agent to break it down into sub-questions.

* **Intermediate Output:** List of sub-questions (received from the Planner).

* **Action 2 - Delegation (Concurrent):** The Manager triggers the Researcher agent for each sub-question concurrently (Concurrent execution) to maximize efficiency and save runtime.

* **Action 3 - Collection & Synthesis:** The Manager passes all raw research outputs to the Aggregator agent to merge them into a unified, deduplicated knowledge base.

* **Action 4 - Hand-off to Writer:** The Manager sends the merged information to the Writer agent.

* **Final Output:** Receives the final report from the Writer, builds the final JSON including the report, total duration, and the `agent_trace` of all actions performed throughout the process.

## 2. Planner / Task Decomposer

**Role:** Analyzes the main research topic and breaks it down into a dynamic number of focused research questions.

* **Input:** Raw research topic from the Manager.

* **Action:** Calls the language model with a System Prompt defining it as a research strategist. It must analyze the topic and divide it into the most appropriate number of sub-topics or questions, without a strict numerical limit. The response is returned in a structured format (Structured Output).

* **Output:** Data array (JSON) containing the list of sub-questions.

## 3. Researcher / Information Gatherer

**Role:** Gathers up-to-date data and facts using independent web search tools, based on Tool Calling.

* **Input:** A specific sub-question from the Manager (e.g., "Describe the AI-based coding tools commonly used by Backend developers today").

* **Action:** Executes an Agentic Loop utilizing the Function Calling capabilities of the language model:

  1. The language model receives the sub-question along with a Tool definition that allows web searching (e.g., Bing Search API).

  2. The model decides autonomously when and how to call the tool, generating the optimal search query.

  3. The code executes the search and returns the results to the LLM.

  4. The model processes the search results and produces a factual, filtered, and focused response.

* **Output:** Text containing only data, facts, and key points (Bullet points or short paragraphs), based on the information retrieved from the web.

## 4. Aggregator / Data Synthesizer

**Role:** Receives information from multiple research sources, filters it, removes duplicates, and consolidates it into a single database or unified raw document.

* **Input:** A list of all research outputs returned from the Researcher agents (passed through the Manager).

* **Action:** Calls the language model with a System Prompt to organize and merge information. The model is asked to identify duplicates among the different answers, find connections between the data, and generate a single concentrated summary of information containing all the relevant facts gathered during the research phase.

* **Output:** A consolidated, deduplicated, and thematically organized information document, returned to the Manager.

## 5. Writer / Output Generator

**Role:** Converts the gathered data into a coherent, readable, and professional document.

* **Input:** The merged information returned from the Aggregator agent.

* **Action:** Calls the language model with a System Prompt of a professional content editor. The prompt requires organizing the information, building a logical structure (introduction, body, conclusion), and maintaining a consistent professional tone.

* **Output:** Continuous and formatted text constituting the Final Report.

## Workflow Step-by-Step

1. An HTTP request is received with a Topic.

2. The system executes the main function of the Manager agent.

3. The Manager calls the Planner agent to get a list of sub-questions. The Planner's Trace is saved.

4. The Manager concurrently (Asynchronously) triggers the Researcher agent for each sub-question. The Researcher autonomously manages calls to the web search tool to gather information. The Trace of each research task is saved separately.

5. Upon completion of the Researchers' work, the Manager passes all answers to the Aggregator agent for data merging and filtering. The Aggregator's Trace is saved.

6. The Manager passes the merged information to the Writer agent. The Writer calls the LLM to draft the report. The Writer's Trace is saved.

7. The Manager takes the report, appends all collected Trace records, calculates the total duration, and returns a complete JSON response.