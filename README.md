# LitLens: AI-Powered Research Assistant

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

LitLens uses a dual-agent architecture that communicates through the Model Context Protocol (MCP):

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
  - Analyzes query to identify key concepts and search terms
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
  - Organizes information into a coherent synthesis
- **Output**: Comprehensive synthesis of research findings
- **Key Value**: Transforms disparate information into cohesive knowledge, identifying patterns that would be missed when reading papers in isolation

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
2. Claude formats the query for the SourceSeeker agent using MCP
3. SourceSeeker searches and returns relevant papers
4. Claude passes these papers to the InsightWeaver agent
5. InsightWeaver performs deep analysis and produces a synthesis
6. Claude presents findings to the user in a conversational format

## Key Technical Components

- **arXiv API Integration**: For accessing computer science and AI research papers
- **Semantic Scholar API**: For broader academic coverage with citation networks
- **Local Storage**: For caching results and maintaining research session state
- **MCP Protocol Implementation**: For standardized agent-to-agent communication
- **LangChain Agents**: For intelligent search term planning and query refinement

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

The enhanced SourceSeeker includes:

1. **LLM-Powered Query Optimization**: Uses an LLM to analyze and improve search terms
2. **Citation Network Exploration**: Finds related papers through citation relationships
3. **Cross-Source Ranking**: Intelligently ranks papers from multiple sources
4. **TL;DR Summaries**: Provides AI-generated summaries when available from Semantic Scholar

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
4. The LitLens Academic Search tool will be available to Claude

## Usage Examples

Here are some examples of how to use LitLens with Claude Desktop:

1. **Multi-Source Research Query**:
   ```
   Can you help me find research papers about transformers in natural language processing?
   ```
   
   Claude will use the SourceSeeker to search across all available sources (arXiv and Semantic Scholar) and present the combined results.

2. **Source-Specific Research**:
   ```
   Find recent papers about large language model hallucinations from arXiv only.
   ```
   
   Claude will use only the arXiv source for this search.

3. **Semantic Scholar with Citation Exploration**:
   ```
   What are the main differences between RLHF and DPO for language model alignment? Use Semantic Scholar and explore citations.
   ```
   
   Claude will search Semantic Scholar and explore the citation networks of top papers.

4. **Combined Source Research with Limited Results**:
   ```
   Find papers about quantum computing applications in cryptography with a maximum of 3 results.
   ```
   
   Claude will search all available sources but limit the combined results to just 3 papers.

---

*LitLens: Bringing research into focus*