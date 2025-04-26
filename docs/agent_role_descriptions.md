# LitLens Agent Role Descriptions

## Agent Architecture

The LitLens system consists of several specialized agents, each with a specific role in the research process:

### SourceSeeker Agent

**Primary Role:** Combined research agent that searches multiple academic sources.

**Input:**
- User query (string)
- Optional parameters:
  - `max_docs`: Maximum number of documents to return per source (default: 10)
  - `sources`: List of sources to search (default: ["arxiv", "semantic_scholar"])
  - `year_filter`: Optional minimum year to filter papers by
  - `is_field_evolution`: Boolean indicating if query is about field evolution/history

**Output:**
- Dictionary containing:
  - Combined search results from multiple sources
  - Source-specific results
  - Search status information
  - Processing metadata

**Key Responsibilities:**
- Coordinates parallel searches across multiple academic databases
- Deduplicates results across sources
- Ranks papers by relevance and citation count
- Handles partial results if a source times out
- Filters papers by year if specified
- Optimizes result presentation based on query type

### ArXiv Agent

**Primary Role:** Specialized search agent for the ArXiv repository.

**Input:**
- Search query (string)
- Maximum documents to return
- Optional model name override

**Output:**
- Dictionary containing:
  - Search results (papers)
  - Search queries used
  - Generated search terms
  - Agent reasoning steps

**Key Responsibilities:**
- Generates multiple related search terms for broader coverage
- Performs searches with original and enhanced query terms
- Deduplicates and ranks results
- Provides direct search fallback for speed
- Applies year filtering when needed

### Semantic Scholar Agent

**Primary Role:** Specialized search agent for the Semantic Scholar database.

**Input:**
- Search query (string)
- Maximum documents to return
- Optional model name override

**Output:**
- Dictionary containing:
  - Search results (papers)
  - Paper IDs with retrieved citations
  - Agent reasoning steps

**Key Responsibilities:**
- Searches Semantic Scholar API for relevant papers
- Retrieves citation information for key papers
- Deduplicates and ranks results by citation count
- Provides rich metadata including TLDRs when available

### InsightWeaver Agent

**Primary Role:** Paper analysis and synthesis agent.

**Input:**
- List of papers from SourceSeeker
- Original user query
- Optional parameters:
  - `synthesis_type`: Type of synthesis to generate (comprehensive, concise, comparative)
  - `max_length`: Maximum length of synthesis
  - `model_name`: Optional override for LLM model

**Output:**
- Dictionary containing:
  - Synthesized analysis addressing the query
  - Critical evaluation of the synthesis
  - Query intent analysis
  - Metadata about the synthesis process

**Key Responsibilities:**
- Analyzes query intent to guide synthesis approach
- Extracts and structures key information from papers
- Generates synthesis prompts based on query intent
- Creates comprehensive, technically accurate research synthesis
- Self-criticizes synthesis to identify limitations
- Recommends the most important papers on the topic

## Agent Interaction Patterns

The LitLens agents work together in the following pattern:

1. **Query Processing:** User submits a research query
2. **Source Seeking:** SourceSeeker agent initiates parallel searches across ArXiv and Semantic Scholar
3. **Result Collection:** SourceSeeker collects, deduplicates, and ranks results from all sources
4. **Insight Weaving:** InsightWeaver analyzes the papers, extracts key information, and generates a synthesis
5. **Response Delivery:** The final synthesis with paper recommendations is returned to the user

## Data Flow

```
User Query → SourceSeeker → [ArXiv Agent, Semantic Scholar Agent] → InsightWeaver → Synthesis Response
```

The system is designed to be modular, allowing for:
- Additional source agents to be easily integrated
- Different synthesis strategies based on query intent
- Fallback mechanisms if any component experiences issues

## Technical Integration

All agents are integrated through the Model Context Protocol (MCP) which enables:
- Asynchronous processing of requests
- Stateful tracking of multi-agent interactions
- Structured communication between agents
- Connection to Claude Desktop and other LLM applications