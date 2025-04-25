"""
ArXiv research agent that helps improve search queries and find academic papers.
"""
import os
from typing import List, Dict, Optional, Union, Any

from langchain_community.retrievers import ArxivRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.agents import tool, AgentExecutor, create_react_agent
from langchain_core.tools import tool as core_tool

# Define a prompt template for planning search terms
SEARCH_TERM_PLANNING_TEMPLATE = """
You are a research assistant tasked with translating user queries into effective search terms for academic papers on arXiv.

User query: {query}

Think about the following:
1. What are the core concepts in this query?
2. What synonyms or related terms might be relevant?
3. What specific technical terminology would yield better results?
4. How can we make the search specific enough to get relevant papers but broad enough to catch important work?

Based on your analysis, provide the BEST search query terms for arXiv that will return the most relevant academic papers.
Only return the search terms, nothing else. Do not include explanations or reasoning.
"""

# Define a prompt template for generating multiple search queries
MULTI_QUERY_PLANNING_TEMPLATE = """
You are a research assistant tasked with generating multiple effective search queries for finding academic papers on arXiv.

User query: {query}

Think about the following:
1. What are the core concepts and terminology in this query?
2. What are different synonyms, related terms, and technical variations of these concepts?
3. What are different subfields or approaches related to this topic?
4. What specific methodologies, algorithms, or frameworks are relevant to this topic?

Based on your analysis, provide 3-5 DIFFERENT search queries that will collectively provide comprehensive coverage of the topic.
Each query should focus on a different aspect, approach, or terminology related to the topic.

Format your response as a comma-separated list of queries. For example:
"machine learning optimization techniques, gradient descent algorithms, stochastic optimization methods, neural network training"

Do not include explanations, numbering, or any other text.
"""


@tool
def search_arxiv_with_terms(query: str, max_docs: int = 5) -> List[Dict]:
    """Search arXiv for academic papers based on the provided search terms."""
    if not query:
        return {"error": "Search terms are required"}

    try:
        retriever = ArxivRetriever(
            top_k_results=max_docs,
            load_max_docs=max_docs,
            doc_content_chars_max=4000
        )
        docs = retriever.get_relevant_documents(query)

        results = []
        for doc in docs:
            results.append({
                "title": doc.metadata.get("Title", ""),
                "authors": doc.metadata.get("Authors", ""),
                "summary": doc.metadata.get("Summary", ""),
                "published": doc.metadata.get("Published", ""),
                "url": f"https://arxiv.org/abs/{doc.metadata.get('Entry_ID', '').split('/')[-1]}",
                "content": doc.page_content
            })

        return results
    except Exception as e:
        return {"error": str(e)}


@tool
def generate_related_search_terms(query: str) -> List[str]:
    """Generate related search terms for an academic topic to expand search coverage.
    
    Use this tool to get multiple search variations for a research topic. For example,
    "computer vision" might generate terms like "image recognition", "CNN", "ViT", etc.
    """
    try:
        # Create the multi-query planner
        llm = get_llm()
        multi_query_planner = create_multi_query_planner(llm)
        query_list = multi_query_planner.invoke({"query": query})
        
        # Split the comma-separated list into individual queries and clean any quotation marks
        search_queries = []
        for q in query_list.split(','):
            # Strip whitespace and remove any quotation marks
            cleaned_query = q.strip().replace('"', '').replace("'", '')
            if cleaned_query:  # Only add non-empty queries
                search_queries.append(cleaned_query)
        return search_queries
    except Exception as e:
        return [f"Error generating search terms: {str(e)}"]


def get_llm(model_name: Optional[str] = None) -> BaseLanguageModel:
    """Get an LLM for the agent based on environment variables or default.

    Args:
        model_name: Override the environment variable model name
    """
    model = model_name or os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
    return ChatOpenAI(model=model)


def create_search_term_planner(llm: Optional[BaseLanguageModel] = None) -> LLMChain:
    """Creates a chain for planning better search terms.

    Args:
        llm: Optional language model to use (will use default if not provided)
    """
    if llm is None:
        llm = get_llm()

    prompt = PromptTemplate.from_template(SEARCH_TERM_PLANNING_TEMPLATE)
    chain = prompt | llm | StrOutputParser()
    return chain


def create_multi_query_planner(llm: Optional[BaseLanguageModel] = None) -> LLMChain:
    """Creates a chain for generating multiple search queries.

    Args:
        llm: Optional language model to use (will use default if not provided)
    """
    if llm is None:
        llm = get_llm()

    prompt = PromptTemplate.from_template(MULTI_QUERY_PLANNING_TEMPLATE)
    chain = prompt | llm | StrOutputParser()
    return chain


def create_arxiv_agent(llm: Optional[BaseLanguageModel] = None) -> AgentExecutor:
    """Create a ReAct agent for searching arxiv.

    Args:
        llm: Optional language model to use (will use default if not provided)
    """
    if llm is None:
        llm = get_llm()

    # Define the prompt
    prompt = PromptTemplate.from_template(
        """You are a research assistant that helps users find relevant academic papers by searching arXiv.
        
        {chat_history}
        
        You have access to the following tools: {tools}
        
        IMPORTANT WORKFLOW: When searching for papers, you should ALWAYS:
        1. First generate multiple related search terms using the generate_related_search_terms tool
        2. Then search arXiv with at least 3 different search terms using search_arxiv_with_terms
           - Start with the original query
           - Then use the most relevant alternative terms generated in step 1
           - Make sure to search using different perspectives/aspects of the topic
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
        - Do not include URLs or references to arXiv
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
    tools = [search_arxiv_with_terms, generate_related_search_terms]
    agent = create_react_agent(llm, tools, prompt)

    # Create the executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,  # Set to True during development to see the agent's reasoning
        return_intermediate_steps=True,
        max_iterations=5,  # Allow enough iterations for multiple searches
        handle_parsing_errors=True  # Handle parsing errors gracefully
    )

    return agent_executor


def deduplicate_results(all_results: List[Dict]) -> List[Dict]:
    """Deduplicate search results based on title and rank by relevance.
    
    Args:
        all_results: List of search result dictionaries
        
    Returns:
        Deduplicated list of search results, ranked by frequency and order
    """
    # Track both unique results and count of appearances
    unique_results = {}
    appearance_count = {}
    appearance_position = {}
    
    for i, result in enumerate(all_results):
        # Use title as the unique identifier
        title = result.get("title", "")
        if not title:
            continue
            
        # If this is the first time we've seen this title
        if title not in unique_results:
            unique_results[title] = result
            appearance_count[title] = 1
            appearance_position[title] = i
        else:
            # Increment the count for duplicates
            appearance_count[title] += 1
    
    # Get the unique results as a list
    result_list = list(unique_results.values())
    
    # Sort the results by:
    # 1. Frequency of appearance (more frequent = higher relevance)
    # 2. Position of first appearance (earlier = higher relevance)
    result_list.sort(
        key=lambda r: (-appearance_count.get(r.get("title", ""), 0), 
                       appearance_position.get(r.get("title", ""), float('inf')))
    )
    
    return result_list


def execute_arxiv_search(
    query: str,
    max_docs: int = 10,
    agent_type: str = None,  # Support direct mode for fallback search
    model_name: Optional[str] = None,
    year_filter: Optional[int] = None
) -> Union[Dict[str, Any], Dict[str, List[Dict]]]:
    """Execute an arXiv search using the ReAct agent or direct search.

    Args:
        query: The search query
        max_docs: Maximum number of documents to return per search term
        agent_type: Search mode - "direct" for fast direct search or any other value for full agent
        model_name: Optional override for the LLM model name
        year_filter: Optional minimum year to filter papers by (for recency filtering)

    Returns:
        Dictionary containing search results and additional info
    """
    if not query:
        return {"error": "Query parameter is required"}

    try:
        # Check for direct search mode (fast fallback option)
        if agent_type == "direct":
            # Direct search without optimization - fastest option
            return _execute_direct_arxiv_search(query, max_docs, year_filter)
            
        # Full agent mode (default)
        llm = get_llm(model_name)
        agent_executor = create_arxiv_agent(llm)
        result = agent_executor.invoke({"input": query})

        # Extract search results and reasoning from intermediate steps
        intermediate_steps = result.get("intermediate_steps", [])
        
        # Collect all search results from all search_arxiv_with_terms calls
        all_search_results = []
        search_queries_used = []
        generated_terms = []
        query_to_results = {}
        
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
                
            # Collect results from search_arxiv_with_terms actions
            if tool_name == "search_arxiv_with_terms" and isinstance(observation, list):
                if tool_input and tool_input not in search_queries_used:
                    search_queries_used.append(tool_input)
                    query_to_results[tool_input] = observation
                    all_search_results.extend(observation)
            
            # Collect search terms from generate_related_search_terms actions
            elif tool_name == "generate_related_search_terms" and isinstance(observation, list):
                generated_terms = observation
        
        # Deduplicate results
        unique_results = deduplicate_results(all_search_results)
        
        # Limit to max_docs total (if needed)
        final_results = unique_results[:max_docs] if len(unique_results) > max_docs else unique_results
        
        # Clean up both lists for proper comparison (removing any quotes that might remain)
        clean_generated_terms = [term.replace('"', '').replace("'", '') for term in generated_terms]
        clean_search_queries = [query.replace('"', '').replace("'", '') for query in search_queries_used]
        
        # Determine which search queries were actually used from the generated terms
        used_generated_terms = []
        for gen_term in clean_generated_terms:
            for search_query in clean_search_queries:
                # Check if the generated term is a substring of the search query or vice versa
                if gen_term in search_query or search_query in gen_term:
                    used_generated_terms.append(gen_term)
                    break
        
        return {
            "search_queries": search_queries_used,
            "generated_terms": generated_terms,
            "used_generated_terms": used_generated_terms,
            "query_results": query_to_results,
            "agent_reasoning": intermediate_steps,
            "summary": result.get("output", ""),
            "results": final_results
        }
    except Exception as e:
        return {"error": str(e)}


def _execute_direct_arxiv_search(query: str, max_docs: int = 10, year_filter: Optional[int] = None) -> Dict[str, Any]:
    """Execute a direct arXiv search without agent optimization - fast mode.
    
    This is used as a fallback when the full search times out.
    """
    try:
        # Create retriever with minimal options for speed
        retriever = ArxivRetriever(
            top_k_results=max_docs,
            load_max_docs=max_docs,
            doc_content_chars_max=4000
        )
        
        # Get documents directly
        docs = retriever.get_relevant_documents(query)
        
        # Format results
        results = []
        for doc in docs:
            # Extract year if available
            year = None
            published = doc.metadata.get("Published", "")
            if published:
                import re
                year_match = re.search(r'\b(19|20)\d{2}\b', published)
                if year_match:
                    year = int(year_match.group(0))
            
            # Apply year filter if specified
            if year_filter is not None and year is not None and year < year_filter:
                # Skip papers that don't meet the year filter
                continue
                
            paper = {
                "title": doc.metadata.get("Title", ""),
                "authors": doc.metadata.get("Authors", ""),
                "summary": doc.metadata.get("Summary", ""),
                "published": published,
                "url": f"https://arxiv.org/abs/{doc.metadata.get('Entry_ID', '').split('/')[-1]}",
                "content": doc.page_content
            }
            
            # Add year if we found it
            if year:
                paper["year"] = year
                
            results.append(paper)
            
        return {
            "search_queries": [query],
            "results": results,
            "source": "arxiv",
            "search_mode": "direct"
        }
    except Exception as e:
        return {"error": f"Direct arXiv search failed: {str(e)}"}
