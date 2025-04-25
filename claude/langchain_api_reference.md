langchain: 0.3.23
Main entrypoint into package.

agents
Classes

agents.agent.AgentExecutor

Agent that is using tools.

agents.agent.AgentOutputParser

Base class for parsing agent output into agent action/finish.

agents.agent.BaseMultiActionAgent

Base Multi Action Agent class.

agents.agent.BaseSingleActionAgent

Base Single Action Agent class.

agents.agent.ExceptionTool

Tool that just returns the query.

agents.agent.MultiActionAgentOutputParser

Base class for parsing agent output into agent actions/finish.

agents.agent.RunnableAgent

Agent powered by Runnables.

agents.agent.RunnableMultiActionAgent

Agent powered by Runnables.

agents.agent_iterator.AgentExecutorIterator(...)

Iterator for AgentExecutor.

agents.agent_toolkits.vectorstore.toolkit.VectorStoreInfo

Information about a VectorStore.

agents.agent_toolkits.vectorstore.toolkit.VectorStoreRouterToolkit

Toolkit for routing between Vector Stores.

agents.agent_toolkits.vectorstore.toolkit.VectorStoreToolkit

Toolkit for interacting with a Vector Store.

agents.chat.output_parser.ChatOutputParser

Output parser for the chat agent.

agents.conversational.output_parser.ConvoOutputParser

Output parser for the conversational agent.

agents.conversational_chat.output_parser.ConvoOutputParser

Output parser for the conversational agent.

agents.mrkl.base.ChainConfig(action_name, ...)

Configuration for a chain to use in MRKL system.

agents.mrkl.output_parser.MRKLOutputParser

MRKL Output parser for the chat agent.

agents.openai_assistant.base.OpenAIAssistantAction

AgentAction with info needed to submit custom tool output to existing run.

agents.openai_assistant.base.OpenAIAssistantFinish

AgentFinish with run and thread metadata.

agents.openai_assistant.base.OpenAIAssistantRunnable

Run an OpenAI Assistant.

agents.openai_functions_agent.agent_token_buffer_memory.AgentTokenBufferMemory

Memory used to save agent output AND intermediate steps.

agents.output_parsers.json.JSONAgentOutputParser

Parses tool invocations and final answers in JSON format.

agents.output_parsers.openai_functions.OpenAIFunctionsAgentOutputParser

Parses a message into agent action/finish.

agents.output_parsers.openai_tools.OpenAIToolsAgentOutputParser

Parses a message into agent actions/finish.

agents.output_parsers.react_json_single_input.ReActJsonSingleInputOutputParser

Parses ReAct-style LLM calls that have a single tool input in json format.

agents.output_parsers.react_single_input.ReActSingleInputOutputParser

Parses ReAct-style LLM calls that have a single tool input.

agents.output_parsers.self_ask.SelfAskOutputParser

Parses self-ask style LLM calls.

agents.output_parsers.tools.ToolAgentAction

Create an AgentAction.

agents.output_parsers.tools.ToolsAgentOutputParser

Parses a message into agent actions/finish.

agents.output_parsers.xml.XMLAgentOutputParser

Parses tool invocations and final answers in XML format.

agents.react.output_parser.ReActOutputParser

Output parser for the ReAct agent.

agents.schema.AgentScratchPadChatPromptTemplate

Chat prompt template for the agent scratchpad.

agents.structured_chat.output_parser.StructuredChatOutputParser

Output parser for the structured chat agent.

agents.structured_chat.output_parser.StructuredChatOutputParserWithRetries

Output parser with retries for the structured chat agent.

agents.tools.InvalidTool

Tool that is run when invalid tool name is encountered by agent.

Functions

agents.agent_toolkits.conversational_retrieval.openai_functions.create_conversational_retrieval_agent(...)

A convenience method for creating a conversational retrieval agent.

agents.format_scratchpad.log.format_log_to_str(...)

Construct the scratchpad that lets the agent continue its thought process.

agents.format_scratchpad.log_to_messages.format_log_to_messages(...)

Construct the scratchpad that lets the agent continue its thought process.

agents.format_scratchpad.openai_functions.format_to_openai_function_messages(...)

Convert (AgentAction, tool output) tuples into FunctionMessages.

agents.format_scratchpad.openai_functions.format_to_openai_functions(...)

Convert (AgentAction, tool output) tuples into FunctionMessages.

agents.format_scratchpad.tools.format_to_tool_messages(...)

Convert (AgentAction, tool output) tuples into ToolMessages.

agents.format_scratchpad.xml.format_xml(...)

Format the intermediate steps as XML.

agents.json_chat.base.create_json_chat_agent(...)

Create an agent that uses JSON to format its logic, build for Chat Models.

agents.openai_functions_agent.base.create_openai_functions_agent(...)

Create an agent that uses OpenAI function calling.

agents.openai_tools.base.create_openai_tools_agent(...)

Create an agent that uses OpenAI tools.

agents.output_parsers.openai_tools.parse_ai_message_to_openai_tool_action(message)

Parse an AI message potentially containing tool_calls.

agents.output_parsers.tools.parse_ai_message_to_tool_action(message)

Parse an AI message potentially containing tool_calls.

agents.react.agent.create_react_agent(llm, ...)

Create an agent that uses ReAct prompting.

agents.self_ask_with_search.base.create_self_ask_with_search_agent(...)

Create an agent that uses self-ask with search prompting.

agents.structured_chat.base.create_structured_chat_agent(...)

Create an agent aimed at supporting tools with multiple inputs.

agents.tool_calling_agent.base.create_tool_calling_agent(...)

Create an agent that uses tools.

agents.utils.validate_tools_single_input(...)

Validate tools for single input.

agents.xml.base.create_xml_agent(llm, tools, ...)

Create an agent that uses XML to format its logic.

Deprecated classes

agents.agent.Agent

agents.agent.LLMSingleActionAgent

agents.agent_types.AgentType(value)

agents.chat.base.ChatAgent

agents.conversational.base.ConversationalAgent

agents.conversational_chat.base.ConversationalChatAgent

agents.mrkl.base.MRKLChain

agents.mrkl.base.ZeroShotAgent

agents.openai_functions_agent.base.OpenAIFunctionsAgent

agents.openai_functions_multi_agent.base.OpenAIMultiFunctionsAgent

agents.react.base.DocstoreExplorer(docstore)

agents.react.base.ReActChain

agents.react.base.ReActDocstoreAgent

agents.react.base.ReActTextWorldAgent

agents.self_ask_with_search.base.SelfAskWithSearchAgent

agents.self_ask_with_search.base.SelfAskWithSearchChain

agents.structured_chat.base.StructuredChatAgent

agents.xml.base.XMLAgent

Deprecated functions

agents.agent_toolkits.vectorstore.base.create_vectorstore_agent(...)

agents.agent_toolkits.vectorstore.base.create_vectorstore_router_agent(...)

agents.initialize.initialize_agent(tools, llm)

agents.loading.load_agent(path, **kwargs)

agents.loading.load_agent_from_config(config)

callbacks
Classes

callbacks.streaming_aiter.AsyncIteratorCallbackHandler()

Callback handler that returns an async iterator.

callbacks.streaming_aiter_final_only.AsyncFinalIteratorCallbackHandler(*)

Callback handler that returns an async iterator.

callbacks.streaming_stdout_final_only.FinalStreamingStdOutCallbackHandler(*)

Callback handler for streaming in agents.

callbacks.tracers.logging.LoggingCallbackHandler(logger)

Tracer that logs via the input Logger.

chains
Classes

chains.base.Chain

Abstract base class for creating structured sequences of calls to components.

chains.combine_documents.base.BaseCombineDocumentsChain

Base interface for chains combining documents.

chains.combine_documents.reduce.AsyncCombineDocsProtocol(...)

Interface for the combine_docs method.

chains.combine_documents.reduce.CombineDocsProtocol(...)

Interface for the combine_docs method.

chains.constitutional_ai.models.ConstitutionalPrinciple

Class for a constitutional principle.

chains.conversational_retrieval.base.BaseConversationalRetrievalChain

Chain for chatting with an index.

chains.conversational_retrieval.base.ChatVectorDBChain

Chain for chatting with a vector database.

chains.conversational_retrieval.base.InputType

Input type for ConversationalRetrievalChain.

chains.elasticsearch_database.base.ElasticsearchDatabaseChain

Chain for interacting with Elasticsearch Database.

chains.flare.base.FlareChain

Chain that combines a retriever, a question generator, and a response generator.

chains.flare.base.QuestionGeneratorChain

Chain that generates questions from uncertain spans.

chains.flare.prompts.FinishedOutputParser

Output parser that checks if the output is finished.

chains.hyde.base.HypotheticalDocumentEmbedder

Generate hypothetical document for query, and then embed that.

chains.moderation.OpenAIModerationChain

Pass input through a moderation endpoint.

chains.natbot.crawler.Crawler()

A crawler for web pages.

chains.natbot.crawler.ElementInViewPort

A typed dictionary containing information about elements in the viewport.

chains.openai_functions.citation_fuzzy_match.FactWithEvidence

Class representing a single statement.

chains.openai_functions.citation_fuzzy_match.QuestionAnswer

A question and its answer as a list of facts each one should have a source.

chains.openai_functions.openapi.SimpleRequestChain

Chain for making a simple request to an API endpoint.

chains.openai_functions.qa_with_structure.AnswerWithSources

An answer to the question, with sources.

chains.prompt_selector.BasePromptSelector

Base class for prompt selectors.

chains.prompt_selector.ConditionalPromptSelector

Prompt collection that goes through conditionals.

chains.qa_with_sources.loading.LoadingCallable(...)

Interface for loading the combine documents chain.

chains.qa_with_sources.retrieval.RetrievalQAWithSourcesChain

Question-answering with sources over an index.

chains.qa_with_sources.vector_db.VectorDBQAWithSourcesChain

Question-answering with sources over a vector database.

chains.query_constructor.base.StructuredQueryOutputParser

Output parser that parses a structured query.

chains.query_constructor.parser.ISO8601Date

A date in ISO 8601 format (YYYY-MM-DD).

chains.query_constructor.parser.ISO8601DateTime

A datetime in ISO 8601 format (YYYY-MM-DDTHH:MM:SS).

chains.query_constructor.parser.QueryTransformer(*args)

Transform a query string into an intermediate representation.

chains.query_constructor.schema.AttributeInfo

Information about a data source attribute.

chains.question_answering.chain.LoadingCallable(...)

Interface for loading the combine documents chain.

chains.router.base.MultiRouteChain

Use a single chain to route an input to one of multiple candidate chains.

chains.router.base.Route(destination, ...)

Create new instance of Route(destination, next_inputs)

chains.router.base.RouterChain

Chain that outputs the name of a destination chain and the inputs to it.

chains.router.embedding_router.EmbeddingRouterChain

Chain that uses embeddings to route between options.

chains.router.llm_router.RouterOutputParser

Parser for output of router chain in the multi-prompt chain.

chains.router.multi_retrieval_qa.MultiRetrievalQAChain

A multi-route chain that uses an LLM router chain to choose amongst retrieval qa chains.

chains.sequential.SequentialChain

Chain where the outputs of one chain feed directly into next.

chains.sequential.SimpleSequentialChain

Simple chain where the outputs of one step feed directly into next.

chains.sql_database.query.SQLInput

Input for a SQL Chain.

chains.sql_database.query.SQLInputWithTables

Input for a SQL Chain.

chains.summarize.chain.LoadingCallable(...)

Interface for loading the combine documents chain.

chains.transform.TransformChain

Chain that transforms the chain output.

Functions

chains.combine_documents.reduce.acollapse_docs(...)

Execute a collapse function on a set of documents and merge their metadatas.

chains.combine_documents.reduce.collapse_docs(...)

Execute a collapse function on a set of documents and merge their metadatas.

chains.combine_documents.reduce.split_list_of_docs(...)

Split Documents into subsets that each meet a cumulative length constraint.

chains.combine_documents.stuff.create_stuff_documents_chain(...)

Create a chain for passing a list of Documents to a model.

chains.example_generator.generate_example(...)

Return another example given a list of examples for a prompt.

chains.history_aware_retriever.create_history_aware_retriever(...)

Create a chain that takes conversation history and returns documents.

chains.openai_functions.citation_fuzzy_match.create_citation_fuzzy_match_runnable(llm)

Create a citation fuzzy match Runnable.

chains.openai_functions.openapi.openapi_spec_to_openai_fn(spec)

Convert a valid OpenAPI spec to the JSON Schema format expected for OpenAI

chains.openai_functions.utils.get_llm_kwargs(...)

Return the kwargs for the LLMChain constructor.

chains.prompt_selector.is_chat_model(llm)

Check if the language model is a chat model.

chains.prompt_selector.is_llm(llm)

Check if the language model is a LLM.

chains.query_constructor.base.construct_examples(...)

Construct examples from input-output pairs.

chains.query_constructor.base.fix_filter_directive(...)

Fix invalid filter directive.

chains.query_constructor.base.get_query_constructor_prompt(...)

Create query construction prompt.

chains.query_constructor.base.load_query_constructor_runnable(...)

Load a query constructor runnable chain.

chains.query_constructor.parser.get_parser([...])

Return a parser for the query language.

chains.retrieval.create_retrieval_chain(...)

Create retrieval chain that retrieves documents and then passes them on.

chains.sql_database.query.create_sql_query_chain(llm, db)

Create a chain that generates SQL queries.

chains.structured_output.base.get_openai_output_parser(...)

Get the appropriate function output parser given the user functions.

chains.summarize.chain.load_summarize_chain(llm)

Load summarizing chain.

Deprecated classes

chains.api.base.APIChain

chains.combine_documents.base.AnalyzeDocumentChain

chains.combine_documents.map_reduce.MapReduceDocumentsChain

chains.combine_documents.map_rerank.MapRerankDocumentsChain

chains.combine_documents.reduce.ReduceDocumentsChain

chains.combine_documents.refine.RefineDocumentsChain

chains.combine_documents.stuff.StuffDocumentsChain

chains.constitutional_ai.base.ConstitutionalChain

chains.conversation.base.ConversationChain

chains.conversational_retrieval.base.ConversationalRetrievalChain

chains.llm.LLMChain

chains.llm_checker.base.LLMCheckerChain

chains.llm_math.base.LLMMathChain

chains.llm_summarization_checker.base.LLMSummarizationCheckerChain

chains.mapreduce.MapReduceChain

chains.natbot.base.NatBotChain

chains.qa_generation.base.QAGenerationChain

chains.qa_with_sources.base.BaseQAWithSourcesChain

chains.qa_with_sources.base.QAWithSourcesChain

chains.retrieval_qa.base.BaseRetrievalQA

chains.retrieval_qa.base.RetrievalQA

chains.retrieval_qa.base.VectorDBQA

chains.router.llm_router.LLMRouterChain

chains.router.multi_prompt.MultiPromptChain

Deprecated functions

chains.loading.load_chain(path, **kwargs)

chains.loading.load_chain_from_config(...)

chains.openai_functions.base.create_openai_fn_chain(...)

chains.openai_functions.base.create_structured_output_chain(...)

chains.openai_functions.citation_fuzzy_match.create_citation_fuzzy_match_chain(llm)

chains.openai_functions.extraction.create_extraction_chain(...)

chains.openai_functions.extraction.create_extraction_chain_pydantic(...)

chains.openai_functions.openapi.get_openapi_chain(spec)

chains.openai_functions.qa_with_structure.create_qa_with_sources_chain(llm)

chains.openai_functions.qa_with_structure.create_qa_with_structure_chain(...)

chains.openai_functions.tagging.create_tagging_chain(...)

chains.openai_functions.tagging.create_tagging_chain_pydantic(...)

chains.openai_tools.extraction.create_extraction_chain_pydantic(...)

chains.qa_with_sources.loading.load_qa_with_sources_chain(llm)

chains.query_constructor.base.load_query_constructor_chain(...)

chains.question_answering.chain.load_qa_chain(llm)

chains.structured_output.base.create_openai_fn_runnable(...)

chains.structured_output.base.create_structured_output_runnable(...)

chat_models
Functions

chat_models.base.init_chat_model()

Initialize a ChatModel from the model name and provider.

embeddings
Classes

embeddings.cache.CacheBackedEmbeddings(...)

Interface for caching results from embedding models.

Functions

embeddings.base.init_embeddings(model, *[, ...])

evaluation
Classes

evaluation.agents.trajectory_eval_chain.TrajectoryEval

A named tuple containing the score and reasoning for a trajectory.

evaluation.agents.trajectory_eval_chain.TrajectoryEvalChain

A chain for evaluating ReAct style agents.

evaluation.agents.trajectory_eval_chain.TrajectoryOutputParser

Trajectory output parser.

evaluation.comparison.eval_chain.LabeledPairwiseStringEvalChain

A chain for comparing two outputs, such as the outputs

evaluation.comparison.eval_chain.PairwiseStringEvalChain

A chain for comparing two outputs, such as the outputs

evaluation.comparison.eval_chain.PairwiseStringResultOutputParser

A parser for the output of the PairwiseStringEvalChain.

evaluation.criteria.eval_chain.Criteria(value)

A Criteria to evaluate.

evaluation.criteria.eval_chain.CriteriaEvalChain

LLM Chain for evaluating runs against criteria.

evaluation.criteria.eval_chain.CriteriaResultOutputParser

A parser for the output of the CriteriaEvalChain.

evaluation.criteria.eval_chain.LabeledCriteriaEvalChain

Criteria evaluation chain that requires references.

evaluation.embedding_distance.base.EmbeddingDistance(value)

Embedding Distance Metric.

evaluation.embedding_distance.base.EmbeddingDistanceEvalChain

Use embedding distances to score semantic difference between a prediction and reference.

evaluation.embedding_distance.base.PairwiseEmbeddingDistanceEvalChain

Use embedding distances to score semantic difference between two predictions.

evaluation.exact_match.base.ExactMatchStringEvaluator(*)

Compute an exact match between the prediction and the reference.

evaluation.parsing.base.JsonEqualityEvaluator([...])

Evaluate whether the prediction is equal to the reference after

evaluation.parsing.base.JsonValidityEvaluator(...)

Evaluate whether the prediction is valid JSON.

evaluation.parsing.json_distance.JsonEditDistanceEvaluator([...])

An evaluator that calculates the edit distance between JSON strings.

evaluation.parsing.json_schema.JsonSchemaEvaluator(...)

An evaluator that validates a JSON prediction against a JSON schema reference.

evaluation.qa.eval_chain.ContextQAEvalChain

LLM Chain for evaluating QA w/o GT based on context

evaluation.qa.eval_chain.CotQAEvalChain

LLM Chain for evaluating QA using chain of thought reasoning.

evaluation.qa.eval_chain.QAEvalChain

LLM Chain for evaluating question answering.

evaluation.qa.generate_chain.QAGenerateChain

LLM Chain for generating examples for question answering.

evaluation.regex_match.base.RegexMatchStringEvaluator(*)

Compute a regex match between the prediction and the reference.

evaluation.schema.AgentTrajectoryEvaluator()

Interface for evaluating agent trajectories.

evaluation.schema.EvaluatorType(value)

The types of the evaluators.

evaluation.schema.LLMEvalChain

A base class for evaluators that use an LLM.

evaluation.schema.PairwiseStringEvaluator()

Compare the output of two models (or two outputs of the same model).

evaluation.schema.StringEvaluator()

Grade, tag, or otherwise evaluate predictions relative to their inputs and/or reference labels.

evaluation.scoring.eval_chain.LabeledScoreStringEvalChain

A chain for scoring the output of a model on a scale of 1-10.

evaluation.scoring.eval_chain.ScoreStringEvalChain

A chain for scoring on a scale of 1-10 the output of a model.

evaluation.scoring.eval_chain.ScoreStringResultOutputParser

A parser for the output of the ScoreStringEvalChain.

evaluation.string_distance.base.PairwiseStringDistanceEvalChain

Compute string edit distances between two predictions.

evaluation.string_distance.base.StringDistance(value)

Distance metric to use.

evaluation.string_distance.base.StringDistanceEvalChain

Compute string distances between the prediction and the reference.

Functions

evaluation.comparison.eval_chain.resolve_pairwise_criteria(...)

Resolve the criteria for the pairwise evaluator.

evaluation.criteria.eval_chain.resolve_criteria(...)

Resolve the criteria to evaluate.

evaluation.loading.load_dataset(uri)

Load a dataset from the LangChainDatasets on HuggingFace.

evaluation.loading.load_evaluator(evaluator, *)

Load the requested evaluation chain specified by a string.

evaluation.loading.load_evaluators(evaluators, *)

Load evaluators specified by a list of evaluator types.

evaluation.scoring.eval_chain.resolve_criteria(...)

Resolve the criteria for the pairwise evaluator.

globals
Functions

globals.get_debug()

Get the value of the debug global setting.

globals.get_llm_cache()

Get the value of the llm_cache global setting.

globals.get_verbose()

Get the value of the verbose global setting.

globals.set_debug(value)

Set a new value for the debug global setting.

globals.set_llm_cache(value)

Set a new LLM cache, overwriting the previous value, if any.

globals.set_verbose(value)

Set a new value for the verbose global setting.

hub
Functions

hub.pull(owner_repo_commit, *[, ...])

Pull an object from the hub and returns it as a LangChain object.

hub.push(repo_full_name, object, *[, ...])

Push an object to the hub and returns the URL it can be viewed at in a browser.

indexes
Classes

indexes.vectorstore.VectorStoreIndexWrapper

Wrapper around a vectorstore for easy access.

indexes.vectorstore.VectorstoreIndexCreator

Logic for creating indexes.

memory
Classes

memory.combined.CombinedMemory

Combining multiple memories' data together.

memory.readonly.ReadOnlySharedMemory

Memory wrapper that is read-only and cannot be changed.

memory.simple.SimpleMemory

Simple memory for storing context or other information that shouldn't ever change between prompts.

memory.vectorstore_token_buffer_memory.ConversationVectorStoreTokenBufferMemory

Conversation chat memory with token limit and vectordb backing.

Functions

memory.utils.get_prompt_input_key(inputs, ...)

Get the prompt input key.

Deprecated classes

memory.buffer.ConversationBufferMemory

memory.buffer.ConversationStringBufferMemory

memory.buffer_window.ConversationBufferWindowMemory

memory.chat_memory.BaseChatMemory

memory.entity.BaseEntityStore

memory.entity.ConversationEntityMemory

memory.entity.InMemoryEntityStore

memory.entity.RedisEntityStore

memory.entity.SQLiteEntityStore

memory.entity.UpstashRedisEntityStore

memory.summary.ConversationSummaryMemory

memory.summary.SummarizerMixin

memory.summary_buffer.ConversationSummaryBufferMemory

memory.token_buffer.ConversationTokenBufferMemory

memory.vectorstore.VectorStoreRetrieverMemory

model_laboratory
Classes

model_laboratory.ModelLaboratory(chains[, names])

A utility to experiment with and compare the performance of different models.

output_parsers
Classes

output_parsers.boolean.BooleanOutputParser

Parse the output of an LLM call to a boolean.

output_parsers.combining.CombiningOutputParser

Combine multiple output parsers into one.

output_parsers.datetime.DatetimeOutputParser

Parse the output of an LLM call to a datetime.

output_parsers.enum.EnumOutputParser

Parse an output that is one of a set of values.

output_parsers.fix.OutputFixingParser

Wrap a parser and try to fix parsing errors.

output_parsers.fix.OutputFixingParserRetryChainInput

output_parsers.pandas_dataframe.PandasDataFrameOutputParser

Parse an output using Pandas DataFrame format.

output_parsers.regex.RegexParser

Parse the output of an LLM call using a regex.

output_parsers.regex_dict.RegexDictParser

Parse the output of an LLM call into a Dictionary using a regex.

output_parsers.retry.RetryOutputParser

Wrap a parser and try to fix parsing errors.

output_parsers.retry.RetryOutputParserRetryChainInput

output_parsers.retry.RetryWithErrorOutputParser

Wrap a parser and try to fix parsing errors.

output_parsers.retry.RetryWithErrorOutputParserRetryChainInput

output_parsers.structured.ResponseSchema

Schema for a response from a structured output parser.

output_parsers.structured.StructuredOutputParser

Parse the output of an LLM call to a structured output.

output_parsers.yaml.YamlOutputParser

Parse YAML output using a pydantic model.

Functions

output_parsers.loading.load_output_parser(config)

Load an output parser.

retrievers
Classes

retrievers.contextual_compression.ContextualCompressionRetriever

Retriever that wraps a base retriever and compresses the results.

retrievers.document_compressors.base.DocumentCompressorPipeline

Document compressor that uses a pipeline of Transformers.

retrievers.document_compressors.chain_extract.LLMChainExtractor

Document compressor that uses an LLM chain to extract the relevant parts of documents.

retrievers.document_compressors.chain_extract.NoOutputParser

Parse outputs that could return a null string of some sort.

retrievers.document_compressors.chain_filter.LLMChainFilter

Filter that drops documents that aren't relevant to the query.

retrievers.document_compressors.cross_encoder.BaseCrossEncoder()

Interface for cross encoder models.

retrievers.document_compressors.cross_encoder_rerank.CrossEncoderReranker

Document compressor that uses CrossEncoder for reranking.

retrievers.document_compressors.embeddings_filter.EmbeddingsFilter

Document compressor that uses embeddings to drop documents unrelated to the query.

retrievers.document_compressors.listwise_rerank.LLMListwiseRerank

Document compressor that uses Zero-Shot Listwise Document Reranking.

retrievers.ensemble.EnsembleRetriever

Retriever that ensembles the multiple retrievers.

retrievers.merger_retriever.MergerRetriever

Retriever that merges the results of multiple retrievers.

retrievers.multi_query.LineListOutputParser

Output parser for a list of lines.

retrievers.multi_query.MultiQueryRetriever

Given a query, use an LLM to write a set of queries.

retrievers.multi_vector.MultiVectorRetriever

Retrieve from a set of multiple embeddings for the same document.

retrievers.multi_vector.SearchType(value)

Enumerator of the types of search to perform.

retrievers.parent_document_retriever.ParentDocumentRetriever

Retrieve small chunks then retrieve their parent documents.

retrievers.re_phraser.RePhraseQueryRetriever

Given a query, use an LLM to re-phrase it.

retrievers.self_query.base.SelfQueryRetriever

Retriever that uses a vector store and an LLM to generate the vector store queries.

retrievers.time_weighted_retriever.TimeWeightedVectorStoreRetriever

Retriever that combines embedding similarity with recency in retrieving values.

Functions

retrievers.document_compressors.chain_extract.default_get_input(...)

Return the compression chain input.

retrievers.document_compressors.chain_filter.default_get_input(...)

Return the compression chain input.

retrievers.ensemble.unique_by_key(iterable, key)

Yield unique elements of an iterable based on a key function.

Deprecated classes

retrievers.document_compressors.cohere_rerank.CohereRerank

runnables
Classes

runnables.hub.HubRunnable

An instance of a runnable stored in the LangChain Hub.

runnables.openai_functions.OpenAIFunction

A function description for ChatOpenAI

runnables.openai_functions.OpenAIFunctionsRouter

A runnable that routes to the selected function.

smith
Classes

smith.evaluation.config.EvalConfig

Configuration for a given run evaluator.

smith.evaluation.config.RunEvalConfig

Configuration for a run evaluation.

smith.evaluation.config.SingleKeyEvalConfig

Configuration for a run evaluator that only requires a single key.

smith.evaluation.progress.ProgressBarCallback(total)

A simple progress bar for the console.

smith.evaluation.runner_utils.ChatModelInput

Input for a chat model.

smith.evaluation.runner_utils.EvalError(...)

Your architecture raised an error.

smith.evaluation.runner_utils.InputFormatError

Raised when the input format is invalid.

smith.evaluation.runner_utils.TestResult

A dictionary of the results of a single test run.

smith.evaluation.string_run_evaluator.ChainStringRunMapper

Extract items to evaluate from the run object from a chain.

smith.evaluation.string_run_evaluator.LLMStringRunMapper

Extract items to evaluate from the run object.

smith.evaluation.string_run_evaluator.StringExampleMapper

Map an example, or row in the dataset, to the inputs of an evaluation.

smith.evaluation.string_run_evaluator.StringRunEvaluatorChain

Evaluate Run and optional examples.

smith.evaluation.string_run_evaluator.StringRunMapper

Extract items to evaluate from the run object.

smith.evaluation.string_run_evaluator.ToolStringRunMapper

Map an input to the tool.

Functions

smith.evaluation.name_generation.random_name()

Generate a random name.

smith.evaluation.runner_utils.arun_on_dataset(...)

Run the Chain or language model on a dataset and store traces to the specified project name.

smith.evaluation.runner_utils.run_on_dataset(...)

Run the Chain or language model on a dataset and store traces to the specified project name.

storage
Classes

storage.encoder_backed.EncoderBackedStore(...)

Wraps a store with key and value encoders/decoders.

storage.file_system.LocalFileStore(root_path, *)

BaseStore interface that works on the local file system.