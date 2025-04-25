"""
Main MCP server for LitLens application.
"""
import os
import logging
import asyncio
import threading
import concurrent.futures
import re
import json
import uuid
import time
from datetime import datetime
import hashlib
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP

from agents.source_seeker_agent import execute_source_seeker_search
from agents.insight_weaver import synthesize_papers

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("litlens")

# Create logs directory if it doesn't exist
logs_dir = Path(__file__).parent.parent / "logs"
logs_dir.mkdir(exist_ok=True)
source_seeker_logs_dir = logs_dir / "source_seeker"
source_seeker_logs_dir.mkdir(exist_ok=True)
insight_weaver_logs_dir = logs_dir / "insight_weaver"
insight_weaver_logs_dir.mkdir(exist_ok=True)

def log_to_file(data: Dict[str, Any], agent_type: str):
    """Log data to a file with UUID and timestamp in the logs directory.
    
    Args:
        data: The data to log
        agent_type: The type of agent (source_seeker or insight_weaver)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_id = str(uuid.uuid4())[:8]
    
    if agent_type.lower() == "source_seeker":
        log_dir = source_seeker_logs_dir
    elif agent_type.lower() == "insight_weaver":
        log_dir = insight_weaver_logs_dir
    else:
        log_dir = logs_dir
    
    log_path = log_dir / f"{timestamp}_{agent_type}_{log_id}.json"
    
    # Add timestamp and request_id to data
    data["_meta"] = {
        "timestamp": timestamp,
        "request_id": log_id,
        "agent_type": agent_type
    }
    
    try:
        with open(log_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Logged {agent_type} data to {log_path}")
    except Exception as e:
        logger.error(f"Error logging to file: {str(e)}")

# Create an MCP server
mcp = FastMCP("LitLens")

# Create a thread pool executor for running async tasks
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Create a lock for thread safety with asyncio operations
loop_lock = threading.Lock()

def analyze_query_constraints(query: str) -> Tuple[Optional[int], bool, Optional[List[str]]]:
    """
    Analyze query to extract temporal constraints, field evolution indicators, and technical domains.
    
    Args:
        query: The search query
        
    Returns:
        Tuple of (year_filter, is_field_evolution_query, technical_domains)
    """
    query_lower = query.lower()
    year_filter = None
    is_field_evolution = False
    technical_domains = []
    
    # Detect technical domains
    # This is a basic pattern matching - could be enhanced with a more sophisticated approach
    technical_domain_patterns = {
        "cryptography": ["cryptography", "encryption", "cipher", "lattice-based", "rlwe", "post-quantum", "homomorphic"],
        "machine_learning": ["deep learning", "neural network", "reinforcement learning", "transformer", "attention mechanism"],
        "quantum_computing": ["quantum", "qubit", "entanglement", "superposition", "shor", "grover"],
        "distributed_systems": ["distributed", "consensus", "blockchain", "byzantine", "peer-to-peer"],
        "computer_vision": ["computer vision", "image processing", "object detection", "segmentation", "cnn"],
        "nlp": ["natural language processing", "nlp", "language model", "sentiment analysis", "translation"]
    }
    
    for domain, keywords in technical_domain_patterns.items():
        if any(keyword in query_lower for keyword in keywords):
            technical_domains.append(domain)
    
    # Check for sub-topic filtering
    subtopic_match = re.search(r'(?:focus(?:ing|ed)? on|only|specifically about|limited to)\s+([^.?!]+)', query_lower)
    if subtopic_match:
        subtopic = subtopic_match.group(1).strip()
        if subtopic and len(subtopic) > 3:  # Ensure it's a meaningful subtopic
            technical_domains.append(f"subtopic:{subtopic}")
    
    # Check for explicit time periods
    explicit_year_match = re.search(r'(\d{4})\s*(-|to)\s*(\d{4}|\bpresent\b)', query_lower)
    if explicit_year_match:
        start_year = int(explicit_year_match.group(1))
        return start_year, is_field_evolution, technical_domains
    
    # Check for "last X years/months"
    last_years_match = re.search(r'last\s+(\d+)\s+(years?|months?)', query_lower)
    if last_years_match:
        time_value = int(last_years_match.group(1))
        time_unit = last_years_match.group(2)
        
        current_year = datetime.now().year
        if 'month' in time_unit:
            # Convert months to fractional years for simplicity
            years_back = time_value / 12
            if years_back < 0.5:  # If less than 6 months, use just the current year
                year_filter = current_year
            else:
                year_filter = current_year - int(years_back)
        else:
            year_filter = current_year - time_value
            
        return year_filter, is_field_evolution, technical_domains
    
    # Check for "recent" mentions - improved to prioritize more recent papers
    recent_terms = {
        'very recent': 1,  # Last year
        'latest': 1,       # Last year
        'newest': 1,       # Last year
        'current': 1,      # Last year
        'cutting edge': 1, # Last year
        'recent': 2,       # Last 2 years
        'new': 2,          # Last 2 years
        'modern': 3,       # Last 3 years
        'state of the art': 2, # Last 2 years
        'state-of-the-art': 2, # Last 2 years
        'last few years': 3    # Last 3 years
    }
    
    # Find the most restrictive (smallest) year filter based on terms
    year_span = 3  # Default
    for term, span in recent_terms.items():
        if term in query_lower and span < year_span:
            year_span = span
    
    if year_span < 3:  # If we found a more restrictive term
        current_year = datetime.now().year
        year_filter = current_year - year_span
    
    # For papers in the last 6-12 months check
    if "last 6 months" in query_lower or "past 6 months" in query_lower:
        current_year = datetime.now().year
        year_filter = current_year  # Only papers from current year
    
    # Check if query is about field evolution
    evolution_terms = ['evolution', 'history', 'development', 'progress', 'advance', 'change', 'trend', 'timeline', 'over time', 'journey', 'milestone']
    if any(term in query_lower for term in evolution_terms):
        is_field_evolution = True
        # For field evolution, we don't want to restrict by year
        year_filter = None
    
    return year_filter, is_field_evolution, technical_domains

@mcp.tool("LitLens")
def research_and_synthesize(
    query: str,
    max_docs: int = 20,  # Increased from 10 to 20
    sources: List[str] = None,
    synthesis_type: str = "comprehensive",
    max_length: int = 1500,  # Increased from 1000 to 1500
    model_name: Optional[str] = None,
    timeout_seconds: int = 120  # Doubled from 60 to 120 for more processing time
) -> Dict[str, Any]:
    """Find, analyze, and synthesize academic research to answer questions.
    
    This tool performs end-to-end academic research:
    1. Searches multiple academic sources for relevant papers
    2. Analyzes individual papers and their relationships
    3. Synthesizes findings into a coherent answer
    4. Critically evaluates the quality and limitations of its synthesis
    
    Use this tool when:
    - The user is asking an academic research question
    - The user wants a comprehensive literature review on a topic
    - The query requires synthesizing information from multiple sources
    - The user wants expert analysis of current research
    
    Examples of good queries:
    - "What are the current approaches to zero-shot learning?"
    - "How has transformer architecture evolved since the original paper?"
    - "What's the evidence for and against using meditation for anxiety?"
    - "Compare different methods for hydrogen storage in fuel cells"
    
    Do NOT use this tool:
    - For simple factual questions that don't require research synthesis
    - For non-academic topics or general knowledge questions
    - When the user wants just a list of papers without synthesis
    - For questions better answered by general knowledge
    
    Args:
        query: The user's research question or topic.
        max_docs: Maximum number of papers to retrieve and analyze (default: 10).
        sources: List of sources to search. Options: "arxiv", "semantic_scholar".
        synthesis_type: Type of synthesis to generate (options: "comprehensive", "concise", "comparative").
        max_length: Maximum length of synthesis in words (default: 1500).
        model_name: Optional override for the LLM model name.
        timeout_seconds: Maximum time in seconds to wait for results (default: 120).
    """
    # Create a request ID for logging
    request_id = str(uuid.uuid4())
    
    # Log the request
    logger.info(f"LitLens processing query: '{query}', synthesis_type: {synthesis_type}, request_id: {request_id}")
    
    # Analyze query for technical domains and time constraints
    year_filter, is_field_evolution, technical_domains = analyze_query_constraints(query)
    
    # Log query analysis
    log_data = {
        "request_id": request_id,
        "query": query,
        "year_filter": year_filter,
        "is_field_evolution": is_field_evolution,
        "technical_domains": technical_domains,
        "max_docs": max_docs,
        "synthesis_type": synthesis_type
    }
    log_to_file(log_data, "litlens_request")
    
    # Phase 1: SourceSeeker - allocate 60% of timeout for search, 40% for synthesis
    search_timeout = int(timeout_seconds * 0.6)
    start_time = time.time()
    
    try:
        # Internal implementation of SourceSeeker
        # Extract technical domain terms to enhance search
        domain_terms = []
        for domain in technical_domains:
            if domain.startswith("subtopic:"):
                domain_terms.append(domain[9:])  # Extract subtopic term
        
        # Enhance query with technical domain terms if applicable
        enhanced_query = query
        if domain_terms:
            # Add technical terms in parentheses to ensure they're included in search
            technical_terms = " ".join(domain_terms)
            if technical_terms not in query:
                enhanced_query = f"{query} ({technical_terms})"
        
        # Direct call to internal function instead of the SourceSeeker tool
        source_seeker_partial_results = {
            "query": enhanced_query,
            "sources_requested": sources if sources else ["arxiv", "semantic_scholar"],
            "sources_completed": [],
            "combined_results": [],
            "status": "initiated",
            "year_filter": year_filter,
            "is_field_evolution": is_field_evolution,
            "technical_domains": technical_domains
        }
        
        source_seeker_future = executor.submit(
            _run_source_seeker_in_thread,
            query=enhanced_query,
            max_docs=max_docs,
            sources=sources,
            model_name=model_name,
            partial_results=source_seeker_partial_results,
            year_filter=year_filter,
            is_field_evolution=is_field_evolution,
            request_id=request_id
        )
        
        # Wait for result with timeout
        try:
            source_seeker_results = source_seeker_future.result(timeout=search_timeout)
            if isinstance(source_seeker_results, dict) and "combined_results" in source_seeker_results:
                papers = source_seeker_results["combined_results"]
            else:
                papers = []
                
            source_seeker_time = time.time() - start_time
            logger.info(f"SourceSeeker completed in {source_seeker_time:.2f}s, found {len(papers)} papers")
            
            # Log SourceSeeker results
            source_seeker_log = {
                "request_id": request_id,
                "query": query,
                "enhanced_query": enhanced_query,
                "year_filter": year_filter,
                "is_field_evolution": is_field_evolution,
                "technical_domains": technical_domains,
                "paper_count": len(papers),
                "execution_time": source_seeker_time,
                "papers": papers
            }
            log_to_file(source_seeker_log, "source_seeker")
            
        except concurrent.futures.TimeoutError:
            logger.warning(f"SourceSeeker timed out after {search_timeout}s")
            source_seeker_future.cancel()
            
            # Use whatever partial results we have
            papers = source_seeker_partial_results["combined_results"]
            source_seeker_results = {
                "status": "timeout",
                "message": f"Search timed out after {search_timeout} seconds",
                "combined_results": papers,
                "sources_completed": source_seeker_partial_results["sources_completed"]
            }
            
            # Log timeout
            log_to_file({
                "request_id": request_id,
                "status": "timeout",
                "partial_papers": len(papers),
                "execution_time": search_timeout
            }, "source_seeker")
        
        # Skip synthesis if no papers found
        if not papers:
            no_results_response = {
                "query": query,
                "papers": [],
                "synthesis": "No relevant papers were found for this query.",
                "evaluation": "Unable to provide analysis due to lack of relevant academic sources."
            }
            log_to_file({**no_results_response, "request_id": request_id}, "insight_weaver")
            return no_results_response
        
        # Phase 2: InsightWeaver - Calculate remaining time for synthesis
        remaining_time = timeout_seconds - (time.time() - start_time)
        if remaining_time < 20:  # Ensure at least 20 seconds for synthesis (increased from 10)
            remaining_time = 20
            
        synthesis_start_time = time.time()
        
        # Create future for synthesis with timeout
        synthesis_future = executor.submit(
            synthesize_papers,
            papers=papers,
            query=query,
            synthesis_type=synthesis_type,
            max_length=max_length,
            model_name=model_name
        )
        
        try:
            synthesis_result = synthesis_future.result(timeout=remaining_time)
            synthesis_time = time.time() - synthesis_start_time
            
            # Log InsightWeaver results
            insight_log = {
                "request_id": request_id,
                "query": query,
                "paper_count": len(papers),
                "synthesis_type": synthesis_type,
                "execution_time": synthesis_time,
                "query_intent": synthesis_result.get("query_intent", {}),
                "synthesis_length": len(synthesis_result.get("synthesis", "")),
                "evaluation_length": len(synthesis_result.get("evaluation", ""))
            }
            log_to_file(insight_log, "insight_weaver")
            
            # Combine everything into final response with both papers and synthesis
            result = {
                "query": query,
                "papers": papers,
                "synthesis": synthesis_result.get("synthesis", ""),
                "evaluation": synthesis_result.get("evaluation", ""),
                "metadata": {
                    "sources_searched": source_seeker_results.get("sources_searched", []),
                    "sources_completed": source_seeker_results.get("sources_completed", []),
                    "paper_count": len(papers),
                    "synthesis_type": synthesis_type,
                    "query_intent": synthesis_result.get("query_intent", {}),
                    "technical_domains": technical_domains,
                    "total_execution_time": time.time() - start_time
                }
            }
            
            # Log final result metadata
            log_to_file({
                "request_id": request_id,
                "total_time": time.time() - start_time,
                "status": "completed",
                "paper_count": len(papers)
            }, "litlens_response")
            
            return result
            
        except concurrent.futures.TimeoutError:
            synthesis_future.cancel()
            
            # If synthesis times out, return papers with a message
            timeout_result = {
                "query": query,
                "papers": papers,
                "synthesis": "The synthesis process timed out. Here are the relevant papers we found.",
                "evaluation": "Unable to complete synthesis within the time limit.",
                "metadata": {
                    "sources_searched": source_seeker_results.get("sources_searched", []),
                    "sources_completed": source_seeker_results.get("sources_completed", []),
                    "paper_count": len(papers),
                    "status": "synthesis_timeout"
                }
            }
            
            log_to_file({
                "request_id": request_id,
                "status": "synthesis_timeout",
                "paper_count": len(papers)
            }, "insight_weaver")
            
            return timeout_result
    
    except Exception as e:
        logger.error(f"Error in research_and_synthesize: {str(e)}")
        
        # Log the error
        log_to_file({
            "request_id": request_id,
            "error": str(e),
            "traceback": str(e.__traceback__)
        }, "litlens_error")
        
        return {
            "error": f"An error occurred while processing your research query: {str(e)}",
            "query": query
        }

# SourceSeeker tool has been removed as a standalone tool.
# All functionality is now integrated into the unified LitLens tool above.


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
    is_field_evolution: bool = False,
    request_id: Optional[str] = None
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