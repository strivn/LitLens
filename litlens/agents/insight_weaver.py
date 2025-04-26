"""
InsightWeaver - Paper analysis and synthesis agent for LitLens.
"""
import os
from typing import List, Dict, Optional, Any, Union
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI

def get_llm(model_name: Optional[str] = None) -> BaseLanguageModel:
    """Get an LLM for the agent based on environment variables or default.

    Args:
        model_name: Override the environment variable model name
    """
    model = model_name or os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
    return ChatOpenAI(model=model)

def analyze_query_intent(query: str) -> Dict[str, Any]:
    """Analyze the intent behind the query to guide synthesis approach.
    
    Args:
        query: The user's query
        
    Returns:
        Dictionary with analysis of query intent
    """
    # Define intent categories
    intent_types = [
        "overview",         # General overview of a topic
        "comparison",       # Comparing approaches or methods
        "state_of_art",     # Current cutting-edge research
        "historical",       # Development of field over time
        "methodology",      # Focus on methods or techniques
        "application",      # Practical applications
        "debate",           # Areas of disagreement in the field
        "evidence",         # Strength of evidence for a claim
        "gap",              # Research gaps or limitations
    ]
    
    # Create pattern matching for intent types
    intent_markers = {
        "overview": ["overview", "introduction", "summary", "explain", "what is"],
        "comparison": ["compare", "difference", "versus", "vs", "contrast", "better"],
        "state_of_art": ["current", "recent", "latest", "state of the art", "cutting edge", "modern"],
        "historical": ["history", "evolution", "development", "over time", "progress", "timeline"],
        "methodology": ["method", "technique", "approach", "implementation", "procedure", "algorithm"],
        "application": ["application", "use case", "practical", "industry", "real-world", "applied"],
        "debate": ["debate", "controversy", "disagreement", "conflicting", "argument"],
        "evidence": ["evidence", "support", "proof", "validation", "empirical", "proven"],
        "gap": ["gap", "limitation", "challenge", "unsolved", "future work", "need for"]
    }
    
    # Analyze query for intents
    query_lower = query.lower()
    detected_intents = {}
    
    for intent, markers in intent_markers.items():
        detected_intents[intent] = any(marker in query_lower for marker in markers)
    
    # Determine primary and secondary intents
    primary_intent = "overview"  # Default
    secondary_intents = []
    
    # Find intents that were detected
    detected = [intent for intent, detected in detected_intents.items() if detected]
    
    if detected:
        primary_intent = detected[0]
        secondary_intents = detected[1:3]  # Take up to 2 secondary intents
    
    return {
        "query": query,
        "primary_intent": primary_intent,
        "secondary_intents": secondary_intents,
        "all_detected_intents": detected
    }

def extract_structured_data(papers: List[Dict], query_intent: Dict[str, Any]) -> List[Dict]:
    """Extract and structure key information from papers based on query intent.
    
    Args:
        papers: List of paper dictionaries from SourceSeeker
        query_intent: Analysis of query intent to guide extraction
        
    Returns:
        List of structured paper data
    """
    structured_papers = []
    
    for paper in papers:
        # Basic information all papers should have
        structured_paper = {
            "title": paper.get("title", ""),
            "authors": paper.get("authors", ""),
            "year": paper.get("year", None),
            "resource_id": paper.get("resource_id", ""),
            "source": paper.get("source", ""),
            "citation_count": paper.get("citation_count", 0),
        }
        
        # Extract summary - prefer TLDR when available for brevity
        summary = paper.get("summary", "")
        if "tldr" in paper and paper["tldr"]:
            structured_paper["key_points"] = paper["tldr"]
            structured_paper["full_summary"] = summary
        else:
            # Try to extract key points from the summary
            structured_paper["full_summary"] = summary
            structured_paper["key_points"] = summary[:250] if len(summary) > 250 else summary
            
        # Add source-specific identifiers for reference
        if "arxiv_id" in paper:
            structured_paper["arxiv_id"] = paper["arxiv_id"]
        if "semantic_scholar_id" in paper:
            structured_paper["semantic_scholar_id"] = paper["semantic_scholar_id"]
            
        structured_papers.append(structured_paper)
    
    # Sort based on query intent
    intent = query_intent.get("primary_intent", "overview")
    
    if intent == "state_of_art" or "recent" in query_intent.get("all_detected_intents", []):
        # For state of art, prioritize recent papers
        structured_papers.sort(key=lambda p: p.get("year", 0), reverse=True)
    elif intent == "historical":
        # For historical queries, sort chronologically
        structured_papers.sort(key=lambda p: p.get("year", 0))
    else:
        # Default sort by citation count (importance)
        structured_papers.sort(key=lambda p: p.get("citation_count", 0), reverse=True)
    
    return structured_papers

def generate_synthesis_prompt(query: str, structured_data: List[Dict], synthesis_type: str, max_length: int) -> str:
    """Generate a prompt for paper synthesis based on the query and available data.
    
    Args:
        query: Original user query
        structured_data: Structured paper information
        synthesis_type: Type of synthesis to generate
        max_length: Maximum length of synthesis
        
    Returns:
        Prompt for LLM to generate synthesis
    """
    # Format paper information for the prompt
    papers_text = ""
    for i, paper in enumerate(structured_data, 1):
        year_str = f" ({paper['year']})" if paper.get('year') else ""
        papers_text += f"\nPAPER {i}: {paper['title']} by {paper['authors']}{year_str}\n"
        papers_text += f"ID: {paper['resource_id']}\n"
        # Include more technical details in the key points
        papers_text += f"KEY POINTS: {paper['key_points']}\n"
        # Add full summary for more context
        if paper.get('full_summary') and paper.get('full_summary') != paper.get('key_points'):
            papers_text += f"FULL SUMMARY: {paper['full_summary'][:1500]}...\n" if len(paper.get('full_summary', '')) > 1500 else f"FULL SUMMARY: {paper['full_summary']}\n"
        
    # Generate appropriate instructions based on synthesis type
    instructions = ""
    if synthesis_type == "comprehensive":
        instructions = """
            Provide a comprehensive synthesis that:
            1. Directly answers the query with technical depth and precision
            2. Summarizes the current state of knowledge with specific technical details
            3. Highlights areas of consensus and disagreement among researchers
            4. Connects findings across multiple papers, noting technical relationships
            5. Identifies strengths and limitations in the methodologies and implementations
            6. Makes specific recommendations for which 3-5 papers are most important to read on this topic
        """
    elif synthesis_type == "concise":
        instructions = """
            Provide a concise synthesis that:
            1. Directly answers the query with technical accuracy and depth
            2. Highlights the most important technical findings from the literature
            3. Focuses only on the strongest evidence and most significant innovations
            4. Recommends the 2-3 most critical papers to read on this specific topic
        """
    elif synthesis_type == "comparative":
        instructions = """
            Provide a comparative synthesis that:
            1. Identifies different technical approaches or methodologies
            2. Directly compares implementation details, results, and technical limitations
            3. Evaluates the relative strengths and weaknesses of each approach using specific metrics
            4. Concludes with which approaches have the strongest technical merit
            5. Recommends the most significant papers for each distinct approach
        """
    else:
        # Default to comprehensive
        instructions = """
            Provide a synthesis that directly answers the query with technical depth and precision.
            Include specific findings, implementation details, and technical considerations.
            Make clear recommendations for the most important papers to read on this topic.
        """
    
    # Additional request for paper recommendations
    recommendation_request = """
    PAPER RECOMMENDATIONS:
    After synthesizing the research, explicitly recommend the 3-5 most important papers that someone interested in this topic should read, explaining why each is significant. Consider factors such as:
    - Technical innovation and contribution
    - Methodological rigor
    - Significance of results
    - Influence on subsequent research
    """
    
    # Construct the full prompt with researcher persona
    prompt = f"""
    You are an expert academic researcher with deep technical knowledge in your field. You adhere to the highest standards of scholarly rigor and technical accuracy. You prioritize precision, evidence-based reasoning, and intellectual honesty in all your work.
    
    USER QUERY: {query}
    
    AVAILABLE PAPERS:
    {papers_text}
    
    SYNTHESIS INSTRUCTIONS:
    {instructions}
    
    {recommendation_request}
    
    RESEARCHER PERSONA GUIDELINES:
    - You are writing for a technically sophisticated audience with domain expertise
    - Use precise technical terminology appropriate to the domain
    - Prioritize depth and accuracy over accessibility to non-experts
    - Be forthright about limitations in the literature or methodological concerns
    - Include specific technical details, metrics, and implementation considerations
    - Apply rigorous critical thinking to evaluate claims and evidence
    - Maintain scholarly objectivity and resist oversimplification
    - Acknowledge gaps in the research and areas requiring further investigation
    
    FORMAT REQUIREMENTS:
    - Keep your synthesis under {max_length} words
    - Start with a direct, technically precise answer to the query
    - Use sections with headings to organize your synthesis
    - Cite papers by their numbers (e.g., [1], [3,4])
    - Be specific about which papers support each technical claim
    - Clearly indicate when you're inferring beyond what's directly stated in the papers
    - Conclude with "Recommended Papers" section listing the most important papers
    
    YOUR SYNTHESIS:
    """
    
    return prompt

def self_criticize_synthesis(query: str, papers: List[Dict], synthesis: str) -> str:
    """Critically evaluate the quality and limitations of the synthesis.
    
    Args:
        query: Original user query
        papers: List of papers used for synthesis
        synthesis: The generated synthesis
        
    Returns:
        Critical evaluation of the synthesis
    """
    # Format minimal paper metadata for the critique
    papers_meta = ""
    for i, paper in enumerate(papers, 1):
        year = paper.get("year", "Unknown year")
        papers_meta += f"[{i}] {paper.get('title', 'Untitled')} ({year}) - {paper.get('authors', 'Unknown authors')}\n"
    
    # Create a prompt for self-criticism
    prompt = f"""
    You are a critical evaluator of academic research synthesis.
    
    ORIGINAL QUERY: {query}
    
    PAPERS USED FOR SYNTHESIS:
    {papers_meta}
    
    SYNTHESIS PROVIDED:
    {synthesis}
    
    Please critically evaluate this synthesis by considering:
    
    1. COVERAGE: What topics, viewpoints, or important papers might be missing?
    2. EVIDENCE STRENGTH: How strong is the evidence supporting the main conclusions?
    3. CONTRADICTIONS: Are there conflicting findings that weren't adequately addressed?
    4. RECENCY: Does the synthesis include the most up-to-date research on this topic?
    5. METHODOLOGY ASSESSMENT: What limitations exist in the methodologies of the cited papers?
    6. INFORMATION GAPS: What specific information is lacking that would strengthen the synthesis?
    
    Provide a concise critical evaluation highlighting limitations and gaps in the synthesis. 
    Format your response in sections with clear headings.
    """
    
    # Use LLM to generate the evaluation
    llm = get_llm()
    evaluation = llm.predict(prompt)
    
    return evaluation

def synthesize_papers(papers: List[Dict], query: str, synthesis_type: str = "comprehensive", max_length: int = 3000, model_name: Optional[str] = None) -> Dict[str, Any]:
    """Generate a synthesis of the papers that addresses the query.
    
    Args:
        papers: List of papers from SourceSeeker
        query: Original user query
        synthesis_type: Type of synthesis to generate
        max_length: Maximum length of synthesis
        model_name: Optional override for LLM model
        
    Returns:
        Dictionary with synthesis and metadata
    """
    # Use a more capable model for synthesis if available
    synthesis_model = model_name or os.environ.get("SYNTHESIS_MODEL_NAME", os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini"))
    llm = get_llm(synthesis_model)
    
    # 1. Analyze query intent
    query_intent = analyze_query_intent(query)
    
    # 2. Extract and structure data from papers
    structured_data = extract_structured_data(papers, query_intent)
    
    # 3. Generate synthesis
    synthesis_prompt = generate_synthesis_prompt(
        query=query,
        structured_data=structured_data,
        synthesis_type=synthesis_type,
        max_length=max_length
    )
    
    synthesis = llm.predict(synthesis_prompt)
    
    # 4. Self-criticize the synthesis
    evaluation = self_criticize_synthesis(
        query=query,
        papers=papers,
        synthesis=synthesis
    )
    
    # 5. Return the complete package
    return {
        "query": query,
        "synthesis": synthesis,
        "evaluation": evaluation,
        "query_intent": query_intent,
        "synthesis_type": synthesis_type,
        "paper_count": len(papers)
    }