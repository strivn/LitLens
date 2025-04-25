"""
SourceSeeker - A combined research agent that searches multiple academic sources.
"""
import os
import re
from typing import List, Dict, Optional, Union, Any
import asyncio

from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI

from .arxiv_agent import execute_arxiv_search
from .semantic_scholar_agent import execute_semantic_scholar_search


def get_llm(model_name: Optional[str] = None) -> BaseLanguageModel:
    """Get an LLM for the agent based on environment variables or default.

    Args:
        model_name: Override the environment variable model name
    """
    model = model_name or os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
    return ChatOpenAI(model=model)


async def execute_source_seeker_search(
    query: str,
    max_docs: int = 10,
    sources: List[str] = None,
    model_name: Optional[str] = None,
    partial_results: Optional[Dict] = None,
    year_filter: Optional[int] = None,
    is_field_evolution: bool = False
) -> Dict[str, Any]:
    """Execute a search across multiple academic sources.

    Args:
        query: The search query
        max_docs: Maximum number of documents to return per source
        sources: List of sources to search (defaults to all available)
        model_name: Optional override for the LLM model name
        partial_results: Optional dictionary to update with partial results as they arrive
        year_filter: Optional minimum year to filter papers by (for recency filtering)
        is_field_evolution: Whether this query is about field evolution/history

    Returns:
        Dictionary containing combined search results and additional info
    """
    if not query:
        return {"error": "Query parameter is required"}
        
    # Set default sources if none provided
    if sources is None:
        sources = ["arxiv", "semantic_scholar"]
    
    # Validate sources
    valid_sources = ["arxiv", "semantic_scholar"]
    for source in sources:
        if source not in valid_sources:
            return {"error": f"Invalid source: {source}. Valid sources are: {', '.join(valid_sources)}"}
    
    # Initialize tracking for partial results
    if partial_results is None:
        partial_results = {
            "query": query,
            "sources_requested": sources,
            "sources_completed": [],
            "combined_results": [],
            "status": "in_progress"
        }
    
    try:
        # Create tasks for each source
        tasks = []
        source_tasks = {}
        
        if "arxiv" in sources:
            # ArXiv search is synchronous, so we need to run it in an executor
            arxiv_task = asyncio.to_thread(
                execute_arxiv_search,
                query=query,
                max_docs=max_docs,
                model_name=model_name
            )
            tasks.append(arxiv_task)
            source_tasks[len(tasks)-1] = "arxiv"
            
        if "semantic_scholar" in sources:
            semantic_task = execute_semantic_scholar_search(
                query=query,
                max_docs=max_docs,
                model_name=model_name
            )
            tasks.append(semantic_task)
            source_tasks[len(tasks)-1] = "semantic_scholar"
        
        # Create as_completed iterator to get results as they arrive
        source_results = {}
        all_papers = []
        errors = []
        
        # Convert tasks to futures
        pending = asyncio.as_completed(tasks)
        
        for i, future in enumerate(pending):
            try:
                # Get the result for this task
                result = await future
                
                # Find the source for this task
                # This is an approximation since as_completed doesn't preserve the original order
                # But it's close enough for our purposes
                source_name = None
                for task_index, name in source_tasks.items():
                    if task_index == i:
                        source_name = name
                        break
                        
                if source_name is None:
                    # If we can't determine the source, try to get it from the result
                    if "source" in result:
                        source_name = result["source"]
                    else:
                        source_name = f"source_{i}"
                
                # Check for errors
                if isinstance(result, Exception):
                    errors.append(f"{source_name}: {str(result)}")
                    continue
                    
                if "error" in result:
                    errors.append(f"{source_name}: {result['error']}")
                    continue
                
                # Add source to papers and filter by year if needed
                result_papers = result.get("results", [])
                for paper in result_papers:
                    paper["source"] = source_name
                    
                    # Apply year filter if specified
                    if year_filter is not None:
                        # Extract year from published date if present
                        paper_year = None
                        if "year" in paper:
                            paper_year = paper["year"]
                        elif "published" in paper:
                            year_match = re.search(r'\b(19|20)\d{2}\b', paper.get("published", ""))
                            if year_match:
                                paper_year = int(year_match.group(0))
                        
                        # Only add papers that meet the year filter
                        if paper_year is not None and paper_year >= year_filter:
                            all_papers.append(paper)
                    else:
                        # No year filter, add all papers
                        all_papers.append(paper)
                    
                # Store the full result
                source_results[source_name] = result
                
                # Update partial results as each source completes
                partial_results["sources_completed"].append(source_name)
                
                # Update the combined results after each source completes
                # This ensures we always have the best available results even if we timeout
                if all_papers:
                    current_papers = deduplicate_cross_source_results(all_papers)
                    limited_papers = current_papers[:max_docs] if len(current_papers) > max_docs else current_papers
                    partial_results["combined_results"] = limited_papers
                
            except Exception as e:
                errors.append(f"Task {i}: {str(e)}")
        
        # All tasks completed - finalize the result
        # Deduplicate and rank papers across all sources
        final_papers = deduplicate_cross_source_results(all_papers)
        
        # Limit to max_docs total
        limited_papers = final_papers[:max_docs] if len(final_papers) > max_docs else final_papers
        
        # Build response
        response = {
            "query": query,
            "sources_searched": sources,
            "sources_completed": partial_results["sources_completed"],
            "source_results": source_results,
            "combined_results": limited_papers,
            "status": "completed",
            "year_filter": year_filter,
            "is_field_evolution": is_field_evolution
        }
        
        # Add errors if any
        if errors:
            response["errors"] = errors
            
        return response
    except Exception as e:
        # In case of exception, see if we have partial results to return
        if partial_results and partial_results.get("combined_results"):
            return {
                "query": query,
                "sources_searched": sources,
                "sources_completed": partial_results["sources_completed"],
                "combined_results": partial_results["combined_results"],
                "status": "error",
                "error": str(e)
            }
        return {"error": str(e), "query": query}


def deduplicate_cross_source_results(all_results: List[Dict]) -> List[Dict]:
    """Deduplicate and rank results across different sources.
    
    Args:
        all_results: List of search result dictionaries from multiple sources
        
    Returns:
        Deduplicated list of search results, ranked by relevance
    """
    # Track unique results by title
    unique_results = {}
    appearance_count = {}
    citation_count = {}
    
    for result in all_results:
        # Use title as the unique identifier
        title = result.get("title", "")
        if not title:
            continue
            
        # Clean the title (lowercase, remove punctuation)
        clean_title = title.lower()
        for char in '.,;:?!-()[]{}\'\"':
            clean_title = clean_title.replace(char, '')
        
        # Check for very similar titles
        found_match = False
        for existing_title in list(unique_results.keys()):
            existing_clean = existing_title.lower()
            for char in '.,;:?!-()[]{}\'\"':
                existing_clean = existing_clean.replace(char, '')
                
            # Check if titles are very similar
            if clean_title in existing_clean or existing_clean in clean_title:
                # Keep the result with more info
                current_result = unique_results[existing_title]
                if len(result.get("summary", "")) > len(current_result.get("summary", "")):
                    unique_results[existing_title] = result
                
                # Update counts
                appearance_count[existing_title] = appearance_count.get(existing_title, 0) + 1
                
                # Take max of citation counts
                current_count = citation_count.get(existing_title, 0)
                new_count = result.get("citation_count", 0)
                citation_count[existing_title] = max(current_count, new_count)
                
                found_match = True
                break
        
        if not found_match:
            # This is a new unique result
            unique_results[title] = result
            appearance_count[title] = 1
            citation_count[title] = result.get("citation_count", 0)
    
    # Get the unique results as a list
    result_list = list(unique_results.values())
    
    # Sort the results by:
    # 1. Frequency of appearance (more frequent = higher relevance)
    # 2. Citation count (higher = more important)
    result_list.sort(
        key=lambda r: (-appearance_count.get(r.get("title", ""), 0), 
                      -citation_count.get(r.get("title", ""), 0))
    )
    
    return result_list