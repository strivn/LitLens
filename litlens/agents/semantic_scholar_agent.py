"""
Semantic Scholar research agent for accessing academic papers.
"""
import os
from typing import List, Dict, Optional, Any, Union
import httpx

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.agents import tool, AgentExecutor, create_react_agent

# Semantic Scholar API endpoint
S2_API_BASE_URL = "https://api.semanticscholar.org/graph/v1"

@tool
async def search_semantic_scholar(query: str, max_docs: int = 5) -> List[Dict]:
    """Search Semantic Scholar for academic papers based on the provided search terms."""
    if not query:
        return {"error": "Search terms are required"}

    try:
        # Prepare API request
        url = f"{S2_API_BASE_URL}/paper/search"
        params = {
            "query": query,
            "limit": max_docs,
            "fields": "title,authors,abstract,venue,year,url,citationCount,tldr"
        }
        
        # Make API request
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            
        if response.status_code != 200:
            return {"error": f"API request failed with status code {response.status_code}: {response.text}"}
            
        data = response.json()
        papers = data.get("data", [])
        
        # Format results
        results = []
        for paper in papers:
            # Extract author names
            authors = []
            for author in paper.get("authors", []):
                authors.append(author.get("name", ""))
                
            # Get TLDR if available
            tldr = ""
            if paper.get("tldr"):
                tldr = paper.get("tldr", {}).get("text", "")
                
            results.append({
                "title": paper.get("title", ""),
                "authors": ", ".join(authors),
                "summary": paper.get("abstract", ""),
                "tldr": tldr,
                "published": f"{paper.get('venue', '')} {paper.get('year', '')}".strip(),
                "url": paper.get("url", ""),
                "citation_count": paper.get("citationCount", 0),
                "paperId": paper.get("paperId", "")
            })
            
        return results
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_paper_citations(paper_id: str, max_citations: int = 5) -> List[Dict]:
    """Get citations for a specific paper from Semantic Scholar."""
    if not paper_id:
        return {"error": "Paper ID is required"}

    try:
        # Prepare API request
        url = f"{S2_API_BASE_URL}/paper/{paper_id}/citations"
        params = {
            "limit": max_citations,
            "fields": "title,authors,abstract,venue,year,url,citationCount,tldr"
        }
        
        # Make API request
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            
        if response.status_code != 200:
            return {"error": f"API request failed with status code {response.status_code}: {response.text}"}
            
        data = response.json()
        citations = data.get("data", [])
        
        # Format results
        results = []
        for citation in citations:
            # Extract the cited paper
            paper = citation.get("citingPaper", {})
            
            # Extract author names
            authors = []
            for author in paper.get("authors", []):
                authors.append(author.get("name", ""))
                
            # Get TLDR if available
            tldr = ""
            if paper.get("tldr"):
                tldr = paper.get("tldr", {}).get("text", "")
                
            results.append({
                "title": paper.get("title", ""),
                "authors": ", ".join(authors),
                "summary": paper.get("abstract", ""),
                "tldr": tldr,
                "published": f"{paper.get('venue', '')} {paper.get('year', '')}".strip(),
                "url": paper.get("url", ""),
                "citation_count": paper.get("citationCount", 0),
                "paperId": paper.get("paperId", "")
            })
            
        return results
    except Exception as e:
        return {"error": str(e)}


def get_llm(model_name: Optional[str] = None) -> BaseLanguageModel:
    """Get an LLM for the agent based on environment variables or default.

    Args:
        model_name: Override the environment variable model name
    """
    model = model_name or os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
    return ChatOpenAI(model=model)


def create_semantic_scholar_agent(llm: Optional[BaseLanguageModel] = None) -> AgentExecutor:
    """Create a ReAct agent for searching Semantic Scholar.

    Args:
        llm: Optional language model to use (will use default if not provided)
    """
    if llm is None:
        llm = get_llm()

    # Define the prompt
    prompt = PromptTemplate.from_template(
        """You are a research assistant that helps users find relevant academic papers by searching Semantic Scholar.
        
        {chat_history}
        
        You have access to the following tools: {tools}
        
        IMPORTANT WORKFLOW: When searching for papers, you should ALWAYS:
        1. First search Semantic Scholar with the user's query
        2. For the most relevant papers found, get their citations to find related work
        3. Combine and analyze the results to provide the most comprehensive coverage of the topic
        
        CRITICAL FORMAT INSTRUCTIONS:
        - Your final answer MUST be structured as a highly concise summary followed by a simple, numbered list of papers
        - LIMIT your response to a brief 1-2 sentence summary followed by listing at most 5 papers
        - Format each paper with only: title, authors, and 1-2 sentence summary
        - Do NOT include lengthy explanations or analysis
        - Do NOT include URLs or links
        - Be extremely brief and concise
        
        Example good format for your final answer:
        "Here are the most relevant papers on quantum computing. 
        1. 'Quantum Computing Applications' by Smith et al. A survey of practical applications in cryptography.
        2. 'Quantum Algorithms' by Jones et al. Focuses on algorithm complexity improvements."
        
        BAD examples to avoid:
        - Do not write long summaries for each paper
        - Do not include detailed explanations of concepts
        - Do not include URLs or references to external sites
        - Do not write lengthy introduction or conclusion sections
        
        Use the following format:
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: a concise 1-2 sentence summary followed by a simple numbered list of the most relevant papers
        
        Human: {input}
        
        {agent_scratchpad}
        """,
        partial_variables={"chat_history": ""}  # Provide default empty value
    )

    # Create the agent with both tools
    tools = [search_semantic_scholar, get_paper_citations]
    agent = create_react_agent(llm, tools, prompt)

    # Create the executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,  # Set to True during development to see the agent's reasoning
        return_intermediate_steps=True,
        max_iterations=5,  # Allow enough iterations for multiple searches and citation retrievals
        handle_parsing_errors=True  # Handle parsing errors gracefully
    )

    return agent_executor


async def execute_semantic_scholar_search(
    query: str,
    max_docs: int = 5,
    model_name: Optional[str] = None
) -> Union[Dict[str, Any], Dict[str, List[Dict]]]:
    """Execute a Semantic Scholar search using the ReAct agent.

    Args:
        query: The search query
        max_docs: Maximum number of documents to return per search
        model_name: Optional override for the LLM model name

    Returns:
        Dictionary containing search results and additional info
    """
    if not query:
        return {"error": "Query parameter is required"}

    try:
        # Create and run the agent
        llm = get_llm(model_name)
        agent_executor = create_semantic_scholar_agent(llm)
        result = await agent_executor.ainvoke({"input": query})

        # Extract search results and reasoning from intermediate steps
        intermediate_steps = result.get("intermediate_steps", [])
        
        # Collect all search results and citation results
        search_results = []
        citation_results = []
        paper_ids_searched = []
        
        for action, observation in intermediate_steps:
            # Extract the tool and its input
            tool_name = None
            tool_input = None
            
            if isinstance(action, dict):
                tool_name = action.get("tool")
                tool_input = action.get("tool_input")
            elif hasattr(action, "tool"):
                tool_name = action.tool
                tool_input = action.tool_input
                
            # Collect direct search results
            if tool_name == "search_semantic_scholar" and isinstance(observation, list):
                search_results.extend(observation)
                
            # Collect citation results
            elif tool_name == "get_paper_citations" and isinstance(observation, list):
                if tool_input not in paper_ids_searched:
                    paper_ids_searched.append(tool_input)
                    citation_results.extend(observation)
        
        # Combine all unique results
        all_results = search_results + citation_results
        unique_results = deduplicate_results(all_results)
        
        # Limit to max_docs total (if needed)
        final_results = unique_results[:max_docs] if len(unique_results) > max_docs else unique_results
        
        return {
            "source": "semantic_scholar",
            "search_query": query,
            "paper_ids_with_citations": paper_ids_searched,
            "agent_reasoning": intermediate_steps,
            "summary": result.get("output", ""),
            "results": final_results
        }
    except Exception as e:
        return {"error": str(e)}


def deduplicate_results(all_results: List[Dict]) -> List[Dict]:
    """Deduplicate search results based on title and rank by relevance.
    
    Args:
        all_results: List of search result dictionaries
        
    Returns:
        Deduplicated list of search results, ranked by citation count and frequency
    """
    # Track both unique results and count of appearances
    unique_results = {}
    appearance_count = {}
    
    for result in all_results:
        # Use title as the unique identifier
        title = result.get("title", "")
        if not title:
            continue
            
        # If this is the first time we've seen this title
        if title not in unique_results:
            unique_results[title] = result
            appearance_count[title] = 1
        else:
            # Increment the count for duplicates
            appearance_count[title] += 1
    
    # Get the unique results as a list
    result_list = list(unique_results.values())
    
    # Sort the results by:
    # 1. Frequency of appearance (more frequent = higher relevance)
    # 2. Citation count (higher = more important)
    result_list.sort(
        key=lambda r: (-appearance_count.get(r.get("title", ""), 0), 
                      -r.get("citation_count", 0))
    )
    
    return result_list