import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain, Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts.prompt import PromptTemplate
from langchain_text_splitters import TokenTextSplitter

load_dotenv()

app = Flask(__name__)

# --- 1. INITIALIZATION ---
graph = Neo4jGraph()

shared_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

transformer = LLMGraphTransformer(
    llm=shared_llm,
    allowed_nodes=["Concept", "Architecture", "Language", "Challenge", "Metric", "Process"],
    allowed_relationships=["USES", "PART_OF", "CHALLENGES", "RESOLVES", "EVALUATES", "MAPS_TO"],
    node_properties=["description"]
)

CYPHER_TEMPLATE = """Task: Generate exactly ONE Cypher statement to query a Knowledge Graph.
Instructions:
1. Use ONLY one MATCH and one RETURN clause.
2. Put the actual search terms from the question directly into the CONTAINS filter.
3. To find relationships, use: MATCH (n)-[r*..2]-(m) WHERE toLower(n.id) CONTAINS 'term1' AND toLower(m.id) CONTAINS 'term2' RETURN n, r, m
4. To find single nodes, use: MATCH (n) WHERE toLower(n.id) CONTAINS 'term1' RETURN n.id, n.description
5. DO NOT use parameters like $entity1. Use the actual words.

Schema:
{schema}

Question: {question}
Cypher Statement:"""

CYPHER_PROMPT = PromptTemplate(input_variables=["schema", "question"], template=CYPHER_TEMPLATE)

# --- 2. API ENDPOINTS ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        nodes = graph.query("MATCH (n) RETURN count(n) AS c")[0]['c']
        rels = graph.query("MATCH ()-[r]->() RETURN count(r) AS c")[0]['c']
        return jsonify({"nodes": nodes, "relationships": rels})
    except:
        return jsonify({"nodes": 0, "relationships": 0})

@app.route('/ingest', methods=['POST'])
def ingest():
    # In a real Flask app, you'd handle file uploads via request.files
    # For now, this triggers the background processing of a path sent via JSON
    data = request.json
    file_path = data.get('path')
    
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "Invalid file path"}), 400

    text_splitter = TokenTextSplitter(chunk_size=2000, chunk_overlap=200)
    loader = PyPDFLoader(file_path)
    chunks = text_splitter.split_documents(loader.load())
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(transformer.convert_to_graph_documents, [c]) for c in chunks]
        for future in futures:
            graph_docs = future.result()
            graph.add_graph_documents(graph_docs, baseEntityLabel=True, include_source=True)

    graph.refresh_schema()
    return jsonify({"status": "Success", "message": f"Mapped {os.path.basename(file_path)}"})

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get('query')
    
    chain = GraphCypherQAChain.from_llm(
        shared_llm, graph=graph, verbose=True, 
        cypher_prompt=CYPHER_PROMPT, allow_dangerous_requests=True
    )
    
    try:
        graph_res = chain.invoke({"query": query})
        graph_output = graph_res['result']
        source = "Knowledge Graph"
        
        # DYNAMIC DEEP DIVE LOGIC
        if len(graph_output.split()) < 25:
            vector_db = Neo4jVector.from_existing_graph(
                embeddings, 
                search_type="hybrid", 
                node_label="Document", 
                text_node_properties=["text"], 
                embedding_node_property="embedding" 
            )
            
            docs = vector_db.similarity_search(query, k=3)
            raw_text = "\n".join([d.page_content for d in docs])
            
            synthesis_prompt = f"""
SYSTEM: You are a Lead Research Scientist. 
INPUT A (Knowledge Graph): {graph_output}
INPUT B (Vector Search): {raw_text}

Combine these into a professional analysis. 
Use Markdown headers, bullet points, and $LaTeX$ for any mathematical or architectural concepts.
Explain clearly how the structural reordering occurs.
"""
            final_res = shared_llm.invoke(synthesis_prompt)
            return jsonify({"answer": final_res.content, "mode": "Deep Dive (Hybrid)"})
        
        return jsonify({"answer": graph_output, "mode": "Graph Search"})
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    graph.query("MATCH (n) DETACH DELETE n")
    return jsonify({"status": "Graph Wiped"})

if __name__ == "__main__":
    app.run(port=5000, debug=True)