import streamlit as st
from langchain_openai import ChatOpenAI
import yaml
import os
from typing import List
from langchain_core.documents import Document
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph

### Index
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_retriever():
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    )
    retriever = vectorstore.as_retriever()
    return retriever

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_retrieval_grander(llm):
    system = """You are a grader assessing relevance
        of a retrieved document to a user question. If the document contains keywords related to the user question,
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
        """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "question: {question}\n\n document: {document} "),
        ]
    )
    retrieval_grader = prompt | llm | JsonOutputParser()
    return retrieval_grader

### Generate
def get_rag_chain(llm):
    system = """You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise.
        if source is exist in context, show the source.
        """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "question: {question}\n\n context: {context} "),
        ]
    )
    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain

### Hallucination Grader
def get_hallucination_grader(llm):
    system = """You are a grader assessing whether
        an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
        single key 'score' and no preamble or explanation."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "documents: {documents}\n\n answer: {generation} "),
        ]
    )

    hallucination_grader = prompt | llm | JsonOutputParser()
    return hallucination_grader

### Answer Grader
def get_answer_grader(llm):
    system = """You are a grader assessing whether an
        answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "question: {question}\n\n answer: {generation} "),
        ]
    )

    answer_grader = prompt | llm | JsonOutputParser()
    return answer_grader

### create OpenAI
def read_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None


yamlData = read_yaml('./key.yaml')
os.environ["OPENAI_API_KEY"] = yamlData['config']['OpenAI']['token']
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

### create Tavily
from tavily import TavilyClient

tavily_token = yamlData['config']['Tavily']['token']
tavily = TavilyClient(api_key=tavily_token)

### get retriever
retriever = get_retriever()

### get retrieval grader
retrieval_grader = get_retrieval_grander(llm)
### Generate
rag_chain = get_rag_chain(llm)

### Hallucination Grader
hallucination_grader = get_hallucination_grader(llm)

### Answer Grader
answer_grader = get_answer_grader(llm)


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]


def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """

    print("--- Docs Retrieval ---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---Relevance Checker---")
    print("--- data count from source : ", len(state["documents"]))
    question = state["question"]
    if state["documents"]:
        documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            filtered_docs.append(d)
        # Document not relevant
        else:
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    web_search = state["web_search"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---NEED WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---GO TO GENERATE---")
        return "generate"


def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    if state["documents"] is None:
        documents = state["documents"]
    else:
        documents = []

    # Web search
    docs = tavily.search(query=question)['results']
    print("---RESULT WEB SEARCH---", len(docs))

    web_results = "\n".join([f"content:{d["content"]} source:{d["url"]}" for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


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

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


# Define the nodes
workflow = StateGraph(GraphState)
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae

workflow.set_entry_point("retrieve")
workflow.add_edge(
    "retrieve",
    "grade_documents"
)

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "grade_documents")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
    },
)

app = workflow.compile()

# Streamlit 앱 UI
st.title("AI Advanced Learning(hook)")
if "messages" not in st.session_state:
    st.session_state["messages"] = []
def add_message():
    user_message = st.session_state["user_input"]
    if user_message:
        st.session_state["messages"].append({"user": "You", "message": user_message})
        st.session_state["user_input"] = ""  # 입력창 초기화
        inputs = {"question": user_message}
        value = app.invoke(inputs)
        # for output in app.stream(inputs):
        #     for key, value in output.items():
        #         print(f"Finished running: {key}:")
        final_report = value["generation"]
        st.session_state["messages"].append({"user": "Bot", "message": final_report})

st.markdown("""
    <style>
        .message {
            margin-bottom: 15px;
        }
        .user-message {
            color: #007bff;
            font-weight: bold;
        }
        .bot-message {
            color: #28a745;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="chat-box">', unsafe_allow_html=True)

for msg in st.session_state["messages"]:
    if msg["user"] == "You":
        st.markdown(f'<div class="message"><span class="user-message">{msg["user"]}:</span> {msg["message"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="message"><span class="bot-message">{msg["user"]}:</span> {msg["message"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.text_input("Enter your message:", key="user_input", on_change=add_message)

if st.button("Send"):
    add_message()