import os
import pickle
import time
import langchain

from typing import List, Dict, Generator, Any
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Import necessary components from the original RAG implementation
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import (
    NVIDIAEmbeddings,
    NVIDIARerank,
    ChatNVIDIA
)
from pydantic import BaseModel, Field

langchain.verbose = False

# Global variables to store initialized components

embeddings = None
reranker = None
llm = None
hybrid_retriever = None
workflow = None
retriever_cache = {}
nearest_jieqi = None


def init_rag_service():
    """Initialize all components needed for the RAG system"""
    global embeddings, reranker, llm, hybrid_retriever, workflow, nearest_jieqi

    # Load environment variables
    load_dotenv()

    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    # Retrieve the API key from the environment
    API_KEY = os.getenv("API_KEY")

    # Validate API key
    if not API_KEY or not API_KEY.startswith("nvapi"):
        raise ValueError("⚠️API KEY must start with nvapi")

    total_start_time = time.time()
    step_start_time = time.time()
    print("---Step-1: Initialize the Embeddings, Reranking and LLM connectors ---")

    # Initialize embeddings, reranker and LLM
    embeddings = NVIDIAEmbeddings(
        model="nvidia/llama-3.2-nv-embedqa-1b-v2",
        api_key=API_KEY,
        truncate="END",
    )

    reranker = NVIDIARerank(
        model="nvidia/llama-3.2-nv-rerankqa-1b-v2",
        api_key=API_KEY,
        truncate="END"
    )

    llm = ChatNVIDIA(
        model="meta/llama-3.3-70b-instruct",
        api_key=API_KEY,
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )

    # Load retrievers
    print("---Step-2: Create a hybrid search retriever ---")
    hybrid_retriever = _setup_hybrid_retriever()

    print("---Step-3: Detect 節氣 ---")
    # Get the current date
    current_date = datetime.now().strftime("%Y-%m-%d")

    # # Query LLM for the nearest 24節氣
    jieqi_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一個24節氣判斷專家，請根據當前日期回答天數最接近的 24 節氣是哪個，僅需兩個字作為答案。"),
        ("human", f"當前日期：{current_date}")
    ])
    jieqi_response = jieqi_prompt | llm | StrOutputParser()
    nearest_jieqi = jieqi_response.invoke({}).strip()
    # nearest_jieqi = "清明"
    print(
        f"---Current Date: {current_date}, Nearest Jieqi: {nearest_jieqi}---")

    print(f"completed in {time.time() - step_start_time:.1f} seconds")
    # Setup workflow graph - 預先設置，無需單獨打印步驟
    workflow = _setup_workflow_graph()

    # Print total initialization time
    print(
        f"Total initialization time: {time.time() - total_start_time:.1f} seconds")
    return True


def load_bm25_retrievers():
    """Load and cache BM25 retrievers"""
    if "bm25" not in retriever_cache:
        bm25_retrievers = []
        for file in os.listdir("./stores"):
            if file.endswith("bm25_retriever.pkl"):
                with open(f"./stores/{file}", "rb") as f:
                    bm25_retrievers.append(pickle.load(f))
        retriever_cache["bm25"] = bm25_retrievers
    return retriever_cache["bm25"]


def load_faiss_retrievers():
    """Load and cache FAISS retrievers"""
    if "faiss" not in retriever_cache:
        faiss_retrievers = []
        for folder in os.listdir("./stores"):
            if folder.endswith("_vectors") and os.path.isdir(f"./stores/{folder}"):
                vectorstore = FAISS.load_local(
                    f"./stores/{folder}", embeddings, allow_dangerous_deserialization=True
                )
                faiss_retrievers.append(
                    vectorstore.as_retriever(search_kwargs={"k": 2}))
        retriever_cache["faiss"] = faiss_retrievers
    return retriever_cache["faiss"]


def _setup_hybrid_retriever():
    """Set up the hybrid retrieval system with caching"""
    # Replace original retriever loading with optimized loading
    bm25_retrievers = load_bm25_retrievers()
    if not bm25_retrievers:
        print("Warning: No BM25 retrievers found.")
        return None

    bm25_retriever = EnsembleRetriever(
        retrievers=bm25_retrievers,
        weights=[1 / len(bm25_retrievers)] * len(bm25_retrievers)
    )

    faiss_retrievers = load_faiss_retrievers()
    if not faiss_retrievers:
        print("Warning: No FAISS retrievers found.")
        return bm25_retriever

    faiss_retriever = EnsembleRetriever(
        retrievers=faiss_retrievers,
        weights=[1 / len(faiss_retrievers)] * len(faiss_retrievers)
    )

    # Create hybrid retriever
    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.7, 0.3]
    )


def _setup_workflow_graph():
    """Set up the LangGraph workflow"""
    # Define state structure
    class GraphState(TypedDict):
        question: str
        sub_questions: List[str]
        generation: str
        documents: List[str]

    # Define structured output for sub-query generation

    class SubQuery(BaseModel):
        """Given a user question related to health supplements (保健食品) or skincare products (保養用品), break it down into distinct sub questions that 
        you need to answer in order to provide a comprehensive and helpful response. If the user's question is vague or incomplete, 
        guide them by generating sub questions that clarify their intent or focus on specific aspects of health supplements or skincare products."""
        questions: List[str] = Field(description="The list of sub questions")

    # Initialize structured generators
    sub_question_generator = llm.with_structured_output(SubQuery)

    # Define the nodes as functions
    def decompose(state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---QUERY DECOMPOSITION ---")
        question = state["question"]

        # Decompose the query into sub-questions
        sub_queries = sub_question_generator.invoke(question)
        unique_sub_questions = list(
            set(sub_queries.questions))  # Remove duplicates

        print(f"---SUB QUERIES: {unique_sub_questions}---")

        # Ensure unique_sub_questions is not empty
        if not unique_sub_questions:
            unique_sub_questions = []

        result = {"sub_questions": unique_sub_questions, "question": question}
        print(result)
        return result

    def retrieve(state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        sub_questions = state["sub_questions"]
        question = state["question"]

        # Parallel retrieval with batching
        documents = []

        def retrieve_for_sub_questions(batch):
            batch_documents = []
            for sub_question in batch:
                batch_documents.extend(hybrid_retriever.invoke(sub_question))
            return batch_documents

        # Batch sub-questions to avoid excessive thread creation
        batch_size = 5  # Adjust batch size based on system resources
        sub_question_batches = [
            sub_questions[i:i + batch_size] for i in range(0, len(sub_questions), batch_size)
        ]

        # Limit threads to avoid overhead
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = executor.map(
                retrieve_for_sub_questions, sub_question_batches)
            for batch_docs in results:
                documents.extend(batch_docs)

        return {"documents": documents, "question": question}

    def rerank(state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RERANK---")
        question = state["question"]
        documents = state["documents"]

        # Reranking
        documents = reranker.compress_documents(
            query=question, documents=documents)
        return {"documents": documents, "question": question}

    def generate(state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # Format documents
        formatted_docs = "\n\n".join(doc.page_content for doc in documents)

        # Use the RAG prompt
        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = prompt | llm | StrOutputParser()

        # Generate the answer
        generation = rag_chain.invoke(
            {"context": formatted_docs, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Define GradeDocuments model
        class GradeDocuments(BaseModel):
            """Binary score for relevance check on retrieved documents."""
            binary_score: str = Field(
                description="Documents are relevant to the question, 'yes' or 'no'"
            )

        # Create grader
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human",
             "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ])

        retrieval_grader = llm.with_structured_output(GradeDocuments)
        retrieval_grader = grade_prompt | retrieval_grader

        # Score each document
        filtered_docs = []
        for d in documents:
            if hasattr(d, 'metadata'):
                print(f"---CHECKING DOCUMENT: {d.metadata}---")
            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content})
            grade = score.binary_score

            print(f"---GRADE: {grade}---")

            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")

        return {"documents": filtered_docs, "question": question}

    def transform_query(state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """
        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Create query rewriter
        system = """You a question re-writer that converts an input question to a better version that is optimized \n 
             for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning. \n
             Given a user question related to health supplements (保健食品) or skincare products (保養用品). \n
             Additionally, consider the current 24節氣: {nearest_jieqi}. If the user's question is vague or incomplete, \n
             clarify their intent or focus on specific aspects of health supplements or skincare products. \n
             Ensure the question are specific, actionable, and cover different dimensions of the user's query, such as benefits, usage, or comparisons"""

        re_write_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human",
             "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ])

        question_rewriter = re_write_prompt | llm | StrOutputParser()
        better_question = question_rewriter.invoke({"question": question})

        return {"documents": documents, "question": better_question}

    # Conditional routing functions
    def decide_to_generate(state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """
        print("---ASSESS GRADED DOCUMENTS---")
        filtered_documents = state["documents"]

        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

    def grade_generation_v_documents_and_question(state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        # Define hallucination grader model
        class GradeHallucinations(BaseModel):
            """Binary score for hallucination present in generation answer."""
            binary_score: str = Field(
                description="Answer is grounded in the facts, 'yes' or 'no'"
            )

        # Create hallucination grader
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
             Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

        hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human",
             "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ])

        hallucination_grader = llm.with_structured_output(GradeHallucinations)
        hallucination_grader = hallucination_prompt | hallucination_grader

        # Grade hallucination
        hallucination_score = hallucination_grader.invoke(
            {"documents": "\n\n".join(
                d.page_content for d in documents), "generation": generation}
        )

        # Check hallucination
        if hallucination_score.binary_score == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")

            # Define answer grader model
            class GradeAnswer(BaseModel):
                """Binary score to assess answer addresses question."""
                binary_score: str = Field(
                    description="Answer addresses the question, 'yes' or 'no'"
                )

            # Create answer grader
            system = """You are a grader assessing whether an answer addresses / resolves a question \n 
                 Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

            answer_prompt = ChatPromptTemplate.from_messages([
                ("system", system),
                ("human",
                 "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ])

            answer_grader = llm.with_structured_output(GradeAnswer)
            answer_grader = answer_prompt | answer_grader

            score = answer_grader.invoke(
                {"question": question, "generation": generation})
            grade = score.binary_score

            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"

            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"

        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

    # Build the graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("decompose", decompose)  # query decompostion
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("rerank", rerank)  # rerank
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("transform_query", transform_query)  # transform_query

    # Build graph
    workflow.add_edge(START, "decompose")
    workflow.add_edge("decompose", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    # Compile
    return workflow.compile()


def process_query(query: str, history: List[Dict[str, str]], context: Dict[str, Any]) -> Generator[str, None, None]:
    """
    Process a user query through the RAG pipeline and yield response chunks

    Args:
        query: The user's question
        history: Conversation history from frontend
        context: Additional context like user_info

    Yields:
        Response text chunks for streaming
    """

    print("===== DEBUGGING =====")
    print(f"Query: {query}")
    print(f"Context: {context}")
    user_info_str = context.get('user_info_str', '')
    print("History items:")
    if history:
        for i, item in enumerate(history):
            print(f"  Item {i}:")
            if isinstance(item, dict):
                for k, v in item.items():
                    print(f"    {k}: {v}")
            else:
                print(f"    {item}")
    else:
        print("  No history items received!")
    print("===== END DEBUGGING =====")

    # Make sure LLM is initialized
    # if not llm:
    #     init_rag_service(user_info_str)

    # First yield the thinking part header
    yield "> **Start thinking**\n\n"

    try:
        # 1. 使用 LLM 判斷問題是否需要產品資料庫
        yield "分析問題類型...\n"

        intent_check = ChatPromptTemplate.from_messages([
            ("system", """你是一個專業的保健保養品問題分析器。請判斷用戶的當前問題是否需要查詢保健保養品資料庫。
            
            特別注意：
            1. 分析對話歷史，識別當前正在討論的健康主題（如睡眠問題、皮膚保養等）
            2. 如果用戶的問題是針對當前主題的跟進提問（如"有其他推薦嗎？"、"哪個品牌好？"等），
            這種情況仍然需要查詢產品資料庫，應回答"需要查詢"
            3. 即使用戶的提問很簡短或不完整，只要它是在延續先前的產品討論，也應判斷為"需要查詢"
            
            例如，如果之前在討論睡眠問題的保健品，用戶問"有其他推薦嗎？"，
            這應該判斷為"需要查詢"，因為用戶是在尋求更多睡眠相關產品的推薦。
            
            請確保考慮完整的對話上下文，找出隱含的主題和意圖。
            只回答："需要查詢" 或 "直接回覆"
            """),
            ("human",
             f"對話歷史：\n{_format_history(history)}\n\n用戶當前問題：{query}\n\n基於對話歷史和當前問題，這是否需要查詢產品資料庫？")
        ])

        intent_result = intent_check | llm | StrOutputParser()
        intent = intent_result.invoke({}).strip()

        # 2. 根據意圖選擇處理路徑
        if "直接回覆" in intent.lower():
            yield "判斷為一般對話，直接回覆...\n"

            direct_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""你是一個專業的保健保養品顧問。請根據對話歷史和當前問題，提供友善、專業的回覆。
                保持對話的自然流暢性。{user_info_str}"""),
                ("human", f"對話歷史：{_format_history(history)}\n\n用戶問題：{query}")
            ])

            direct_response = direct_prompt | llm | StrOutputParser()
            answer = direct_response.invoke({})

            yield f"\n> **End thinking**\n\n{answer}"
            return

        # 3. 執行 RAG 流程
        yield "判斷需要產品資訊，開始進行檢索...\n"

        # # 確保完整初始化
        # if not hybrid_retriever or not workflow:
        #     init_rag_service(user_info_str)

        # 設置循環監控
        recursion_count = 0
        max_attempts = 5
        final_state = None

        # 執行 RAG 工作流
        try:
            inputs = {"question": query}
            for output in workflow.stream(inputs, config={"recursion_limit": 10}):
                current_node = list(output.keys())[0]

                # 監控潛在循環
                if current_node == "transform_query":
                    recursion_count += 1
                    if recursion_count >= max_attempts:
                        print("檢測到潛在循環，切換至直接回覆模式")
                        break

                yield f"執行 {current_node}...\n"
                final_state = output

            # 處理 RAG 結果
            if final_state and recursion_count < max_attempts:
                final_output = list(final_state.values())[-1]
                generation = final_output.get("generation", "")

                if generation and len(generation.strip()) > 10:
                    yield f"\n> **End thinking**\n\n{generation}"
                    return

            # RAG 流程無結果或出現循環，使用 LLM 作為後備
            yield "資料庫未找到相關產品，使用一般知識回覆...\n"

            fallback_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""你是一個專業的保健保養品顧問。用戶詢問的是關於保健保養品的問題，但資料庫中沒有找到特定答案。
                請基於你的專業知識，提供有幫助的一般建議。{user_info_str}"""),
                ("human", f"對話歷史：{_format_history(history)}\n\n用戶問題：{query}")
            ])

            fallback_response = fallback_prompt | llm | StrOutputParser()
            fallback_answer = fallback_response.invoke({})

            yield f"\n> **End thinking**\n\n{fallback_answer}"

        except Exception as rag_error:
            print(f"RAG 流程出錯: {str(rag_error)}")

            # RAG 出錯，使用 LLM 直接回覆
            yield "處理過程出錯，切換到一般回覆模式...\n"

            error_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""你是一個專業的保健保養品顧問。請基於你的知識，回答用戶的問題。
                提供清晰、有用的建議。{user_info_str}"""),
                ("human", query)
            ])

            error_response = error_prompt | llm | StrOutputParser()
            answer = error_response.invoke({})

            yield f"\n> **End thinking**\n\n{answer}"

    except Exception as e:
        print(f"整體處理出錯: {str(e)}")
        import traceback
        traceback.print_exc()

        yield f"\n> **End thinking**\n\n抱歉，處理您的請求時發生錯誤。請嘗試重新描述您的問題，或詢問另一個關於保健保養品的問題。"

# 輔助函數：格式化歷史訊息


def _format_history(history):
    """將歷史記錄格式化為易於 LLM 理解的文本格式"""
    if not history:
        return "無歷史對話"

    formatted = []
    for i, msg in enumerate(history):
        if i % 2 == 0:  # 用戶訊息
            formatted.append(f"用戶: {msg.get('content', '')}")
        else:  # 助手訊息
            formatted.append(f"助手: {msg.get('content', '')}")

    return "\n".join(formatted)

# 輔助函數：格式化用戶資訊


def _format_user_info(user_info):
    """將用戶資訊格式化為字符串"""
    if not user_info:
        return ""

    parts = []
    if user_info.get("gender"):
        parts.append(f"性別: {user_info['gender']}")
    if user_info.get("age"):
        parts.append(f"年齡: {user_info['age']}")
    if user_info.get("zodiac"):
        parts.append(f"星座: {user_info['zodiac']}")
    if user_info.get("mbti"):
        parts.append(f"MBTI: {user_info['mbti']}")

    if parts:
        return "並依據使用者個性 (性別、星座、年齡、MBTI) 調整回答的情緒，用戶資訊：" + ", ".join(parts)
    return ""


# For direct testing
if __name__ == "__main__":
    init_rag_service()
    question = "常長痘痘應該怎麼辦，有什麼保養品可以改善？<節氣：清明, 年齡：20-30, 性別：女>"
    for chunk in process_query(question, [], {}):
        print(chunk, end="", flush=True)
