"""
Main MCP server for LitLens application.
"""
import os
import logging
from typing import Dict, List, Optional, Union, Any
from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP

from agents.arxiv_agent import execute_arxiv_search

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("litlens")

# Create an MCP server
mcp = FastMCP("LitLens")

@mcp.tool()
def search_arxiv(
    query: str, 
    max_docs: int = 5, 
    model_name: Optional[str] = None
) -> Union[Dict[str, Any], Dict[str, List[Dict]]]:
    """Search arXiv for academic papers only when the user wants to find or research scientific/academic publications.
    
    Use this tool when:
    - The user is specifically asking to find academic papers or research
    - The user wants to explore scientific literature on a topic
    - The query is clearly about academic research
    
    Do NOT use this tool:
    - For general information requests where scientific papers aren't specifically requested
    - When the user just mentions "arxiv" in passing
    - For non-academic queries
    
    Args:
        query: The search query for finding relevant academic papers.
        max_docs: Maximum number of documents to return (default: 5).
        model_name: Optional override for the LLM model name.
    """
    logger.info(f"Executing arXiv search with query: '{query}', max_docs: {max_docs}")
    results = execute_arxiv_search(
        query=query, 
        max_docs=max_docs,
        model_name=model_name
    )
    logger.info(f"Search completed. Found {len(results.get('results', [])) if isinstance(results.get('results', []), list) else 0} results")
    return results

if __name__ == "__main__":
    mcp.run(transport='stdio')