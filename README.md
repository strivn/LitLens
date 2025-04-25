# LitLens: AI-Powered Research Assistant


## Disclaimer
All of the code and documentation in this repository was generated with Claude Code assistance. 
They have not been throughly vetted. 


## Overview

LitLens is an intelligent research assistant system that leverages AI agents to help users efficiently find, analyze, and synthesize academic research. The system employs a multi-agent architecture with specialized components that work together to provide comprehensive research assistance with minimal user intervention.

## Why LitLens Exists

Academic research is increasingly overwhelming:

- The volume of published research continues to grow exponentially
- Finding relevant papers across multiple repositories is time-consuming
- Reading and synthesizing information from multiple papers is mentally taxing
- Important connections between papers are often missed
- Context limitations in LLMs make it difficult to process multiple papers at once

LitLens addresses these challenges by creating a specialized agent system that can handle each part of the research process efficiently, delivering concise, well-structured insights without overwhelming the user with information.

## System Architecture

LitLens uses a multi-agent architecture that communicates through the Model Context Protocol (MCP):

## Why We Use Specialized Agents

LitLens deliberately separates research functionality into two specialized agents rather than using a single all-purpose agent:

### Retriever-Synthesizer Separation Benefits

1. **Context Window Optimization**: 
   - Research papers are lengthy and numerous
   - A single agent would quickly exceed context limitations
   - Separate agents can process more total content than would fit in a single context window

2. **Specialized Expertise**:
   - Different skills are needed for search vs. deep analysis
   - The Retriever optimizes for breadth and efficient filtering
   - The Synthesizer optimizes for depth and connection-making

3. **Parallel Processing**:
   - The Synthesizer can begin analysis while the Retriever continues searching
   - This creates a more responsive, efficient research pipeline

4. **Progressive Refinement**:
   - The Synthesizer can request additional sources from the Retriever if gaps are identified
   - This creates an interactive research loop not possible with a single agent

5. **Modularity and Extensibility**:
   - Each agent can be independently improved or replaced
   - Domain-specific versions can be swapped in for specialized fields

### 1. SourceSeeker (Retriever Agent)

The SourceSeeker specializes in finding and initially evaluating relevant research:

- **Input**: User's research query
- **Process**:
  - Analyzes query to identify key concepts, search terms, and constraints
  - Detects technical domains to improve search relevance
  - Extracts temporal constraints for better year filtering
  - Searches multiple academic sources (arXiv, Semantic Scholar, etc.)
  - Performs initial relevance assessment based on abstracts and metadata
  - Ranks and filters papers by relevance score
- **Output**: Collection of potentially relevant papers with metadata
- **Key Value**: Efficiently filters the vast ocean of research down to the most promising candidates

### 2. InsightWeaver (Synthesizer Agent)

The InsightWeaver performs deep analysis of papers identified by the SourceSeeker:

- **Input**: Collection of papers from the SourceSeeker
- **Process**:
  - Reads full papers or targeted sections
  - Groups papers by subtopic and research approach
  - Identifies agreements, contradictions, and trends across papers
  - Extracts key methodologies and findings
  - Adopts an academic researcher persona for rigorous analysis
  - Organizes information into a coherent synthesis
  - Identifies the most important papers with rationale for recommendations
- **Output**: Comprehensive synthesis of research findings with specific paper recommendations
- **Key Value**: Transforms disparate information into cohesive knowledge, identifying patterns that would be missed when reading papers in isolation, and providing guidance on which papers deserve priority attention

### 3. Orchestrator

The Orchestrator manages communication between agents and the user interface:

- Routes messages between components using MCP
- Maintains conversation state
- Handles error conditions and recovery
- Manages authentication and permissions

### 4. Claude Desktop Integration

The system integrates with Claude Desktop as a set of extensions:

- Claude serves as the conversational interface for the user
- The research agents appear as tools/extensions that Claude can call upon
- The system handles passing information between Claude and the specialized agents

## Communication Flow

1. User submits a research query to Claude
2. Claude formats the query for the LitLens system using MCP
3. LitLens analyzes the query to extract constraints, domains, and subtopics
4. SourceSeeker component searches and returns relevant papers
5. InsightWeaver component performs deep analysis and produces a synthesis with recommendations
6. The system logs the entire transaction with a unique identifier
7. Claude presents findings to the user in a conversational format

## Key Technical Components

- **arXiv API Integration**: For accessing computer science and AI research papers
- **Semantic Scholar API**: For broader academic coverage with citation networks
- **Local Storage**: For caching results and maintaining research session state
- **MCP Protocol Implementation**: For standardized agent-to-agent communication
- **LangChain Agents**: For intelligent search term planning and query refinement
- **Structured Logging System**: JSON logs with UUIDs and timestamps for tracking all transactions
- **Technical Domain Patterns**: Pattern matching for specialized academic fields to improve search relevance
- **Query Intent Analysis**: Extraction of constraints, domains, and subtopics from natural language
- **Async/Await Framework**: Proper event loop management with ThreadPoolExecutor for concurrent tasks

## SourceSeeker Multi-Source Search

LitLens SourceSeeker now searches across multiple academic sources:

1. **arXiv**: For computer science, physics, mathematics, and related fields 
2. **Semantic Scholar**: For broader coverage across all academic disciplines with citation data

The SourceSeeker agent intelligently combines and ranks results from these sources, providing:
- Deduplication across sources
- Ranking based on relevance, citation count, and frequency of appearance
- Citation-based discovery for finding related papers

You can control which sources are searched using the `sources` parameter:
- `sources=["arxiv"]` - Search only arXiv
- `sources=["semantic_scholar"]` - Search only Semantic Scholar
- Default (omitted) - Search all available sources

## Advanced Features

LitLens includes the following advanced capabilities:

### Intelligent Search and Analysis

1. **LLM-Powered Query Optimization**: Uses an LLM to analyze and improve search terms
2. **Citation Network Exploration**: Finds related papers through citation relationships
3. **Cross-Source Ranking**: Intelligently ranks papers from multiple sources
4. **TL;DR Summaries**: Provides AI-generated summaries when available from Semantic Scholar
5. **Technical Domain Detection**: Automatically identifies specialized academic fields (e.g., cryptography, machine learning) to improve search relevance
6. **Year Filtering**: Enhanced handling of temporal constraints for accessing recent research in fast-moving fields
7. **Subtopic Extraction**: Ability to filter by specific sub-topics within broader research domains

### InsightWeaver Synthesis

1. **Academic Researcher Persona**: Adopts the rigorous standards and writing style of a professional researcher when synthesizing information
2. **Paper Recommendations**: Identifies the 3-5 most important papers that researchers should prioritize based on technical innovation, methodological rigor, and influence
3. **Pattern Recognition**: Identifies agreements, contradictions, and research trends across multiple papers
4. **Methodological Analysis**: Extracts and compares key methodologies and approaches used across papers

### System Features

1. **Comprehensive Logging**: Structured JSON logs with timestamps and UUIDs for tracking all requests and responses
2. **Unified Tool Interface**: Consolidated interface through the LitLens tool while preserving internal separation of concerns
3. **Asynchronous Processing**: Uses async/await patterns with proper event loop management for responsive performance
4. **Concurrent Task Management**: Employs ThreadPoolExecutor with timeouts to prevent hanging during paper processing

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/litlens.git
   cd litlens
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   # Using pip
   pip install -e .
   
   # Using uv (recommended)
   uv pip install -e .
   ```

4. Install development tools:
   ```bash
   pip install ruff mypy
   # or
   uv pip install ruff mypy
   ```

5. Configure your environment:
   ```bash
   cp .env.example .env
   # Edit .env to add your OpenAI API key
   ```

### Running the Server

Start the MCP server:
```bash
python -m litlens.main
```

The server will start and be available to MCP clients.

### Connecting with Claude Desktop

1. In Claude Desktop, go to Settings > Tools
2. Click "Add MCP Config File" and select the `mcp_config.json` file from this repository
3. Start the server using the command above
4. The LitLens research assistant tool will be available to Claude

### Logging and Debugging

LitLens includes a comprehensive logging system that captures all transactions:

1. **Log Structure**: Each request generates a timestamped JSON log with a unique UUID
2. **Log Location**: Logs are stored in the `/logs` directory with subdirectories for each component
3. **Log Content**: Logs include the original query, detected constraints, papers found, and synthesis results
4. **Debugging**: When troubleshooting, check the logs to see exactly how queries were processed

## Usage Examples

Here are some examples of how to use LitLens with Claude Desktop:

1. **Multi-Source Research Query with Synthesis**:
   ```
   Can you help me find and synthesize research papers about transformers in natural language processing?
   ```
   
   Claude will use LitLens to search across all available sources (arXiv and Semantic Scholar), analyze the papers, and provide a comprehensive synthesis with paper recommendations.

2. **Source-Specific Research with Year Filtering**:
   ```
   Find papers about large language model hallucinations published since 2022 from arXiv only.
   ```
   
   Claude will use only the arXiv source for this search and apply temporal filtering to focus on recent research.

3. **Technical Domain-Specific Research**:
   ```
   What are the main differences between RLHF and DPO for language model alignment? Focus on machine learning papers that explore both approaches.
   ```
   
   LitLens will detect the technical domain (machine learning) and focus the search on papers comparing these alignment techniques.

4. **Research with Subtopic Filtering**:
   ```
   Find papers about quantum computing applications in post-quantum cryptography with a focus on lattice-based approaches.
   ```
   
   LitLens will detect the technical domain (cryptography), the general topic (quantum computing), and the specific subtopic (lattice-based approaches) to provide highly relevant results.

5. **Request for Paper Recommendations**:
   ```
   What are the most important papers I should read about diffusion models in computer vision?
   ```
   
   LitLens will not only search for papers but will provide specific recommendations of the most important papers with explanations of why each is significant.

6. **Access to Recent Research in Fast-Moving Field**:
   ```
   What are the latest developments in multimodal large language models in the last 6 months?
   ```
   
   LitLens will detect the temporal constraint and focus on very recent papers in this rapidly evolving field.

---

*LitLens: Bringing research into focus*