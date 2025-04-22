from langchain_community.retrievers import ArxivRetriever
from mcp.server.fastmcp import FastMCP

# Create an MCP server
app = FastMCP("LitLens")

@app.tool()
def search_arxiv(query: str, max_docs: int = 5) -> list:
    """Search arXiv for academic papers based on the provided query."""
    if not query:
        return {"error": "Query parameter is required"}
    
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
        
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    app.run(transport='stdio')