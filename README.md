🔬 Dynamic GraphRAG Research Assistant




An autonomous research agent that bridges the gap between Linguistic Constraints and Architectural Solutions. This system uses a Knowledge Graph to map high-level concepts and a Vector Deep Dive mechanism to extract technical details from complex research modules.

🌟 Features
Hybrid GraphRAG: Combines the structural reasoning of a Knowledge Graph (Neo4j) with the semantic depth of Vector Search.

Autonomous Deep Dive: The agent self-evaluates its answers. If a relationship in the graph is found but lacks detail, the system automatically triggers a "Deep Dive" into the raw PDF text.

Gemini 2.0 Flash Powered: High-speed, high-accuracy reasoning and synthesis of technical research papers.

Linguistic Mapping: Specifically tuned to identify relationships between language typologies (SVO, SOV) and neural architectures (Transformers, Encoder-Decoder).

Flask Web Interface: A modern, dark-themed dashboard with real-time graph statistics and reasoning logs.

🏗️ The Architecture
The system follows a three-stage reasoning process:

Graph Mapping: Ingests PDFs and converts them into a network of nodes (Concepts, Architecture, Challenges) and relationships (USES, RESOLVES, MAPS_TO).

Pathfinding: When a question is asked, the agent queries the graph to find connections between entities.

Vector Fallback (The Deep Dive): If the graph answer is sparse, the agent performs a similarity search on text embeddings to provide a detailed technical explanation.

🚀 Getting Started
1. Prerequisites
Python 3.9+

Neo4j Database (Local or AuraDB)

Google Gemini API Key

2. Installation
Bash

# Clone the repository
git clone https://github.com/yourusername/dynamic-graphrag.git
cd dynamic-graphrag

# Install dependencies
pip install -r requirements.txt
3. Configuration
Create a .env file in the root directory:

Code snippet

GOOGLE_API_KEY=your_gemini_api_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
4. Running the App
Bash

python main.py
Visit http://127.0.0.1:5000 in your browser.

📊 Example Research Discovery
Question: "How does the Encoder-Decoder model handle SVO reordering?"

Analysis:

Graph Result: Finds that Encoder-Decoder Architecture MENTIONS SVO Language.

Deep Dive Triggered: Since the link is sparse, the agent scans the PDF chunks.

Final Synthesis: Explains how Cross-Attention allows the Decoder to selectively focus on the Encoder's hidden states to reorder words from Subject-Verb-Object into the target language's typology.

🛠️ Built With
LangChain - Orchestration of LLM and Graph chains.

Neo4j - Graph database for structured knowledge.

Gemini 2.0 Flash - Core reasoning and text generation.

Flask - Web backend.

Bootstrap 5 - Frontend UI.
