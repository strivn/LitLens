# LitLens Sample Interaction Transcripts

This document provides sample interaction transcripts showing the multi-agent exchanges within the LitLens system. These examples demonstrate how the different specialized agents work together to process research queries.

## Sample Interaction #1: Researching Agentic AI Risks

### User Query
```
Autonomous AI agents risks research autonomy alignment safety recent
```

![Request Flow](../assets/request_flow.png)

### Process Flow

#### 1. Request Initialization
```json
{
  "request_id": "6b259a49-b40c-4bf3-9efc-f644166d8685",
  "query": "Autonomous AI agents risks research autonomy alignment safety recent",
  "year_filter": 2023,
  "is_field_evolution": false,
  "technical_domains": [],
  "max_docs": 15,
  "synthesis_type": "comprehensive",
  "_meta": {
    "timestamp": "20250425_214811",
    "request_id": "715352ad",
    "agent_type": "litlens_request"
  }
}
```

#### 2. SourceSeeker Agent Activation

The SourceSeeker agent processes the request and initiates parallel searches to ArXiv and Semantic Scholar.

```json
{
  "request_id": "6b259a49-b40c-4bf3-9efc-f644166d8685",
  "query": "Autonomous AI agents risks research autonomy alignment safety recent",
  "enhanced_query": "Autonomous AI agents risks research autonomy alignment safety recent",
  "year_filter": 2023,
  "is_field_evolution": false,
  "technical_domains": [],
  "paper_count": 5,
  "execution_time": 9.403462886810303
}
```

![Agent Message Flow](../assets/agent_message_flow.png)

The agent successfully retrieves relevant papers from both sources, including:

1. "Prioritizing Safeguarding Over Autonomy: Risks of LLM Agents for Science"
2. "Designing for Human-Agent Alignment: Understanding what humans want from their agents"
3. "CERN for AI: a theoretical framework for autonomous simulation-based artificial intelligence testing and alignment"
4. "Fully Autonomous AI Agents Should Not be Developed"
5. "Autonomous Alignment with Human Value on Altruism through Considerate Self-imagination and Theory of Mind"

#### 3. InsightWeaver Agent Processing

![SourceSeeker to InsightWeaver](../assets/sourceseeker_to_insightweaver.png)

The InsightWeaver agent receives the papers and begins the synthesis process:

```json
{
  "request_id": "6b259a49-b40c-4bf3-9efc-f644166d8685",
  "query": "Autonomous AI agents risks research autonomy alignment safety recent",
  "paper_count": 5,
  "synthesis_type": "comprehensive",
  "execution_time": 52.855950117111206,
  "query_intent": {
    "query": "Autonomous AI agents risks research autonomy alignment safety recent",
    "primary_intent": "state_of_art",
    "secondary_intents": [],
    "all_detected_intents": [
      "state_of_art"
    ]
  }
}
```

#### 4. Final Response

The system returns a comprehensive synthesis:

```json
{
  "request_id": "6b259a49-b40c-4bf3-9efc-f644166d8685",
  "total_time": 62.26197004318237,
  "status": "completed",
  "paper_count": 5,
  "_meta": {
    "timestamp": "20250425_214913",
    "request_id": "6a4ef1f7",
    "agent_type": "litlens_response"
  }
}
```

## Sample Interaction #2: Field Evolution Research

### User Query
```
Trace the evolution of computer vision algorithms from traditional methods to deep learning approaches
```

### Process Flow

#### 1. Request Initialization
```json
{
  "request_id": "8c3f5d21-7a94-4e15-b8d3-92a514fe3c22",
  "query": "Trace the evolution of computer vision algorithms from traditional methods to deep learning approaches",
  "year_filter": null,
  "is_field_evolution": true,
  "technical_domains": [
    "computer_vision",
    "machine_learning"
  ],
  "max_docs": 15,
  "synthesis_type": "comprehensive"
}
```

#### 2. SourceSeeker Agent Activation

The SourceSeeker detects this is a field evolution query and adjusts its strategy:
- Sets chronological sorting instead of citation-based ranking
- Searches broadly across time periods
- Focuses on review papers and surveys

```json
{
  "query": "Trace the evolution of computer vision algorithms from traditional methods to deep learning approaches",
  "sources_requested": ["arxiv", "semantic_scholar"],
  "is_field_evolution": true,
  "paper_count": 15,
  "execution_time": 22.34789276123047
}
```

#### 3. ArXiv Agent Processing

The ArXiv agent generates specialized search terms:
```
"history of computer vision algorithms", "evolution of image processing techniques", "computer vision before deep learning", "transition from traditional computer vision to CNN", "survey of computer vision historical development"
```

#### 4. InsightWeaver Agent Processing

The InsightWeaver agent:
- Identifies query intent as "historical"
- Sorts papers chronologically
- Structures synthesis to trace algorithm evolution
- Creates timeline of key developments
- Highlights paradigm shifts in the field

#### 5. Final Synthesis

The InsightWeaver produces a comprehensive synthesis that:
- Traces the evolution from early edge detection to modern neural architectures
- Organizes content by decades and key paradigm shifts
- Compares traditional feature engineering with learned representations
- Provides technical explanations of algorithm progression
- Recommends seminal papers that marked turning points in the field

### Key Agent Interactions

Throughout this exchange, the agents demonstrate several important interactions:
1. SourceSeeker customizes search strategy based on field_evolution flag
2. ArXiv agent generates historically-oriented search terms
3. InsightWeaver adjusts synthesis structure for historical narrative
4. Results are presented chronologically rather than by citation impact
5. Final synthesis connects papers across different time periods to show evolution

This sample demonstrates how the system adapts to different query types, with agents communicating contextual information to provide the most appropriate response.

## Communication Protocol

In both examples, the agents communicate through structured JSON messages that include:

1. **Request Context**: Query parameters, flags, and metadata
2. **Search Results**: Papers with full metadata from each source
3. **Processing Status**: Completion status, timing information, error handling
4. **Agent Reasoning**: Tracking of agent decision process and intermediate steps

This protocol enables asynchronous processing, error recovery, and coordinated multi-agent responses.