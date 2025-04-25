"""
Main MCP server for LitLens application.
"""
import os
import logging
import asyncio
import threading
import concurrent.futures
import re
from datetime import datetime
import hashlib
from typing import Dict, List, Optional, Union, Any, Tuple
from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP

from agents.source_seeker_agent import execute_source_seeker_search

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("litlens")

# Create an MCP server
mcp = FastMCP("LitLens")

# Create a thread pool executor for running async tasks
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Create a lock for thread safety with asyncio operations
loop_lock = threading.Lock()

def analyze_query_constraints(query: str) -> Tuple[Optional[int], bool]:
    """
    Analyze query to extract temporal constraints and detect if it's about field evolution.
    
    Args:
        query: The search query
        
    Returns:
        Tuple of (year_filter, is_field_evolution_query)
    """
    query_lower = query.lower()
    year_filter = None
    is_field_evolution = False
    
    # Check for explicit time periods
    explicit_year_match = re.search(r'(\d{4})\s*(-|to)\s*(\d{4}|\bpresent\b)', query_lower)
    if explicit_year_match:
        start_year = int(explicit_year_match.group(1))
        return start_year, is_field_evolution
    
    # Check for "last X years"
    last_years_match = re.search(r'last\s+(\d+)\s+years', query_lower)
    if last_years_match:
        years_back = int(last_years_match.group(1))
        current_year = datetime.now().year
        year_filter = current_year - years_back
        return year_filter, is_field_evolution
    
    # Check for "recent" mentions
    recent_terms = ['recent', 'latest', 'new', 'current', 'modern', 'last few years', 'state of the art', 'state-of-the-art', 'cutting edge']
    if any(term in query_lower for term in recent_terms):
        current_year = datetime.now().year
        year_filter = current_year - 3  # Default to last 3 years for "recent"
    
    # Check if query is about field evolution
    evolution_terms = ['evolution', 'history', 'development', 'progress', 'advance', 'change', 'trend', 'timeline', 'over time', 'journey', 'milestone']
    if any(term in query_lower for term in evolution_terms):
        is_field_evolution = True
        # For field evolution, we don't want to restrict by year
        year_filter = None
    
    return year_filter, is_field_evolution

@mcp.tool("SourceSeeker")
def search_academic_sources(
    query: str, 
    max_docs: int = 10,
    sources: List[str] = None,
    model_name: Optional[str] = None,
    timeout_seconds: int = 30
) -> Union[Dict[str, Any], Dict[str, List[Dict]]]:
    """Find and analyze academic research papers from multiple scholarly sources.
    
    Use this tool when:
    - The user is specifically asking to find academic papers or research
    - The user wants to explore scientific literature on a technical topic
    - The query is clearly about academic or scholarly research
    - The question involves technical concepts that would benefit from academic research insights
    
    Examples of good queries:
    - "Find papers about transformer architectures in NLP"
    - "What are recent developments in quantum computing algorithms?"
    - "Research on blockchain consensus mechanisms"
    - "Find academic papers about climate modeling techniques"
    
    Do NOT use this tool:
    - For general information requests where scientific papers aren't specifically requested
    - For non-academic queries or general knowledge questions
    - For broad topics without technical specificity (e.g., "blockchain trends")
    - For questions better answered by general knowledge
    
    IMPORTANT USAGE GUIDANCE:
    - Pass the user's raw query exactly as entered (e.g., "what are recent developments in ViT?")
    - Do NOT attempt to extract keywords or reformulate the query
    - The system will automatically convert natural language questions into optimized search terms
    - Passing the complete question gives the internal LLMs more context for generating better search terms
    - The system handles question-style queries just as well as keyword queries
    
    Args:
        query: The user's raw search query or question, passed exactly as entered.
        max_docs: Maximum number of documents to return (default: 10).
        sources: List of sources to search. Options: "arxiv", "semantic_scholar". 
                 If not provided, searches all available sources.
        model_name: Optional override for the LLM model name.
        timeout_seconds: Maximum time in seconds to wait for results (default: 30).
    """
    # Analyze query for temporal constraints and field evolution
    year_filter, is_field_evolution = analyze_query_constraints(query)
    
    logger.info(f"SourceSeeker executing search with query: '{query}', max_docs: {max_docs}, sources: {sources}, timeout: {timeout_seconds}s, year_filter: {year_filter}, field_evolution: {is_field_evolution}")
    
    # For storing partial results in case of timeout
    partial_results = {
        "query": query,
        "sources_requested": sources if sources else ["arxiv", "semantic_scholar"],
        "sources_completed": [],
        "combined_results": [],
        "status": "initiated",
        "year_filter": year_filter,
        "is_field_evolution": is_field_evolution
    }
    
    # Use a simpler approach with a global executor
    try:
        # Create a future for our search
        search_future = executor.submit(
            _run_source_seeker_in_thread,
            query=query,
            max_docs=max_docs,
            sources=sources,
            model_name=model_name,
            partial_results=partial_results,
            year_filter=year_filter,
            is_field_evolution=is_field_evolution
        )
        
        # Wait for the result with a timeout
        try:
            results = search_future.result(timeout=timeout_seconds)
            logger.info(f"SourceSeeker completed within timeout of {timeout_seconds}s")
            # Return simplified results for client, but keep full data internally
            return validate_and_format_response(results, simplify_for_client=True)
        except concurrent.futures.TimeoutError:
            # If we timeout, cancel the future if possible and return partial results
            logger.warning(f"SourceSeeker timed out after {timeout_seconds}s, returning partial results")
            search_future.cancel()
            
            # Update status
            partial_results["status"] = "timeout"
            partial_results["message"] = f"Search timed out after {timeout_seconds} seconds"
            
            # If we have at least some results, it's better than nothing
            if len(partial_results["combined_results"]) > 0:
                logger.info(f"Returning {len(partial_results['combined_results'])} partial results")
                # Validate and simplify partial results before returning
                validated_partial = validate_and_format_response(partial_results, simplify_for_client=True)
                return validated_partial
            else:
                # Otherwise, try a fallback to just arxiv which may be faster
                logger.info("No partial results available, trying fast fallback to arXiv")
                try:
                    from agents.arxiv_agent import execute_arxiv_search
                    fallback_results = execute_arxiv_search(
                        query=query,
                        max_docs=max_docs,
                        model_name=model_name,
                        agent_type="direct",  # Use direct mode for speed
                        year_filter=year_filter  # Pass the year filter to the fallback
                    )
                    
                    # Format as a partial result
                    if "results" in fallback_results and len(fallback_results["results"]) > 0:
                        partial_results["combined_results"] = fallback_results["results"]
                        partial_results["sources_completed"] = ["arxiv_direct"]
                        partial_results["message"] += ". Fallback to direct arXiv search successful."
                        
                        # Validate and simplify the fallback results
                        validated_results = validate_and_format_response(partial_results, simplify_for_client=True)
                        return validated_results
                    else:
                        # If no results, provide a helpful message
                        error_response = {
                            "query": query,
                            "status": "no_results",
                            "message": f"Search timed out after {timeout_seconds} seconds and no relevant papers were found. Try modifying your search terms.",
                            "combined_results": []
                        }
                        return validate_and_format_response(error_response, simplify_for_client=True)
                except Exception as fallback_error:
                    logger.error(f"Fallback search failed: {str(fallback_error)}")
                    error_response = {
                        "error": f"Search timed out after {timeout_seconds} seconds and fallback search failed: {str(fallback_error)}",
                        "query": query,
                        "status": "error",
                        "results": []
                    }
                    return validate_and_format_response(error_response, simplify_for_client=True)
    except Exception as e:
        logger.error(f"SourceSeeker error: {str(e)}")
        error_response = {
            "error": str(e),
            "query": query,
            "status": "error",
            "results": []
        }
        return validate_and_format_response(error_response, simplify_for_client=True)


def validate_and_format_response(results: Dict[str, Any], simplify_for_client: bool = False) -> Dict[str, Any]:
    """Validate and format the search results to ensure they're clean and consistent.
    
    Args:
        results: The search results to validate and format
        simplify_for_client: If True, return only the summary and results (default: False)
    """
    try:
        # Handle error case
        if "error" in results:
            return results
            
        # Process combined results
        if "combined_results" in results:
            # Safety check: if we got no results but the status is "completed", provide a helpful message
            if len(results["combined_results"]) == 0 and results.get("status") == "completed":
                results["message"] = "No relevant papers were found for this query. Try broadening your search terms."
            
            # Ensure summaries aren't too long
            for paper in results["combined_results"]:
                # Basic validation - ensure required fields exist
                if "title" not in paper:
                    paper["title"] = "Untitled Paper"
                if "authors" not in paper:
                    paper["authors"] = "Unknown Authors"
                if "summary" not in paper:
                    paper["summary"] = "No summary available."
                
                # Fix known error patterns in titles
                paper["title"] = paper["title"].replace("\\u2019", "'").replace("\\n", " ")
                
                # Extract and add resource identifiers
                # Add arXiv ID if it's an arXiv paper
                if "url" in paper and "arxiv.org/abs/" in paper.get("url", ""):
                    paper["arxiv_id"] = paper["url"].split("arxiv.org/abs/")[-1]
                    paper["resource_id"] = f"arXiv:{paper['arxiv_id']}"
                elif "Entry_ID" in paper.get("metadata", {}):
                    paper["arxiv_id"] = paper["metadata"]["Entry_ID"].split("/")[-1]
                    paper["resource_id"] = f"arXiv:{paper['arxiv_id']}"
                # Add Semantic Scholar ID if it's from Semantic Scholar
                elif "paperId" in paper:
                    paper["semantic_scholar_id"] = paper["paperId"]
                    paper["resource_id"] = f"S2:{paper['paperId']}"
                # Add DOI if available
                if "doi" in paper:
                    paper["resource_id"] = f"DOI:{paper['doi']}"
                
                # Ensure there's a resource_id even if we couldn't find a specific one
                if "resource_id" not in paper:
                    # Create a resource ID from the title as fallback
                    paper["resource_id"] = f"paper:{hashlib.md5(paper['title'].encode()).hexdigest()[:8]}"
                
                # Add year for sorting/filtering if not present
                if "year" not in paper and "published" in paper:
                    # Try to extract year from the published field
                    year_match = re.search(r'\b(19|20)\d{2}\b', paper.get("published", ""))
                    if year_match:
                        paper["year"] = int(year_match.group(0))
                
                # Limit summary to ~1000 chars if it's too long
                if len(paper.get("summary", "")) > 1000:
                    paper["summary"] = paper["summary"][:997] + "..."
                    
                # Add TLDR from Semantic Scholar if available and summary is long
                if "tldr" in paper and paper.get("tldr") and len(paper.get("summary", "")) > 500:
                    # Clean up TLDR
                    cleaned_tldr = paper["tldr"].replace("\\u2019", "'").replace("\\n", " ")
                    paper["summary"] = f"TLDR: {cleaned_tldr}\n\n{paper['summary']}"
        
            # If this is a field evolution query, rerank the papers to include seminal papers
            if results.get("is_field_evolution", False):
                rerank_for_field_evolution(results)
        
        # Remove large intermediate data to reduce response size
        if "source_results" in results:
            for source, source_data in results["source_results"].items():
                # Remove large arrays that aren't needed by the client
                if "agent_reasoning" in source_data:
                    del source_data["agent_reasoning"]
                    
                # Clean up any summary field in the source data
                if "summary" in source_data:
                    # Fix known error patterns in summaries
                    source_data["summary"] = source_data["summary"].replace("\\u2019", "'").replace("\\n", " ")
        
        # For client responses, simplify to just the essential data
        if simplify_for_client:
            # Preserve the full results internally
            full_results = results.copy()
            
            # Create a simplified version with just summary and results
            simplified = {
                "summary": results.get("summary", ""),
                "results": results.get("combined_results", [])
            }
            
            # Add error message if there is one
            if "error" in results:
                simplified["error"] = results["error"]
            
            # Add helpful message if there is one
            if "message" in results:
                simplified["message"] = results["message"]
                
            # Store the full results in a non-exposed field for future use
            simplified["_full_data"] = full_results
            
            return simplified
            
        return results
    except Exception as e:
        logger.error(f"Error validating response: {str(e)}")
        # Return the original results if validation fails
        return results


def rerank_for_field_evolution(results: Dict[str, Any]) -> None:
    """Rerank papers for field evolution queries to include seminal and recent papers."""
    if "combined_results" not in results or len(results["combined_results"]) < 3:
        return  # Not enough papers to rerank
    
    papers = results["combined_results"]
    
    # Sort by citation count to find seminal papers
    citation_ranked = sorted(papers, key=lambda p: p.get("citation_count", 0), reverse=True)
    
    # Sort by year to find papers across time periods
    # Filter out papers without a year
    papers_with_year = [p for p in papers if "year" in p]
    year_ranked = sorted(papers_with_year, key=lambda p: p.get("year", 0))
    
    # Create a mix of high-citation papers and papers from different time periods
    reranked = []
    
    # Add top cited papers (seminal papers)
    top_cited = citation_ranked[:3]
    for paper in top_cited:
        if paper not in reranked:
            reranked.append(paper)
    
    # Add papers from different time periods if we have year data
    if year_ranked:
        # Get the min and max years
        min_year = year_ranked[0].get("year", 0)
        max_year = year_ranked[-1].get("year", 0)
        
        # Add papers from early, middle, and recent periods
        if max_year > min_year:
            periods = 3  # Early, middle, late
            period_size = (max_year - min_year) / periods
            
            for i in range(periods):
                period_start = min_year + (i * period_size)
                period_end = period_start + period_size
                
                # Find papers in this period
                period_papers = [p for p in year_ranked if p.get("year", 0) >= period_start and p.get("year", 0) < period_end]
                
                # Add the highest cited paper from this period
                if period_papers:
                    period_papers.sort(key=lambda p: p.get("citation_count", 0), reverse=True)
                    if period_papers[0] not in reranked:
                        reranked.append(period_papers[0])
    
    # Add any remaining papers to fill up to the original count
    remaining = [p for p in papers if p not in reranked]
    reranked.extend(remaining[:len(papers) - len(reranked)])
    
    # Update the combined results
    results["combined_results"] = reranked


def _run_source_seeker_in_thread(
    query: str,
    max_docs: int,
    sources: List[str],
    model_name: Optional[str],
    partial_results: Dict,
    year_filter: Optional[int] = None,
    is_field_evolution: bool = False
) -> Dict[str, Any]:
    """Run the SourceSeeker search in a separate thread with proper asyncio handling."""
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Run the coroutine in this thread's event loop
        results = loop.run_until_complete(
            execute_source_seeker_search(
                query=query,
                max_docs=max_docs,
                sources=sources,
                model_name=model_name,
                partial_results=partial_results,
                year_filter=year_filter,
                is_field_evolution=is_field_evolution
            )
        )
        
        # Validate and format the response
        results = validate_and_format_response(results)
        
        # Log results
        if "error" in results:
            logger.error(f"SourceSeeker error: {results['error']}")
        else:
            combined_results = results.get("combined_results", [])
            logger.info(f"SourceSeeker completed. Found {len(combined_results)} combined results")
        
        return results
    except Exception as e:
        logger.error(f"SourceSeeker error during execution: {str(e)}")
        return {"error": str(e), "query": query}
    finally:
        # Clean up properly
        if loop.is_running():
            loop.stop()
        loop.close()

if __name__ == "__main__":
    mcp.run(transport='stdio')