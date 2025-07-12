from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_pinecone import PineconeVectorStore

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel as LCBaseModel, Field
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START
import os
from uuid import uuid4
from langchain_core.documents import Document
from pinecone import Pinecone
from dotenv import load_dotenv

# FastAPI app
app = FastAPI(title="CRAG API", description="Corrective RAG API with Azure OpenAI")

# Load environment variables from .env file
load_dotenv()

# Configuration
# Environment variables are now loaded from .env
PINECONE_INDEX_NAME = "crag-index"

# Initialize Pinecone
pc = Pinecone()
# if PINECONE_INDEX_NAME not in pc.list_indexes():
#     pc.create_index(name=PINECONE_INDEX_NAME, spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}, dimension=1536)

# Make urls global and mutable
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

def clear_vectorstore(vectorstore):
    # Delete all vectors in the namespace
    vectorstore._index.delete(
        namespace="default",
        delete_all=True
    )

def init_retriever():
    global urls
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        namespace="default"
    )

    # Add split documents to the vector store
    uuids = [str(uuid4()) for _ in range(len(doc_splits))]
    vectorstore.add_documents(documents=doc_splits, ids=uuids)

    return vectorstore.as_retriever(), vectorstore

# Initialize LLMs and tools
class GradeDocuments(LCBaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

def init_components():
    # LLM with function call
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Grader prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])

    retrieval_grader = grade_prompt | structured_llm_grader

    # RAG components
    prompt = hub.pull("rlm/rag-prompt")
    llm_gen = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    rag_chain = prompt | llm_gen | StrOutputParser()

    # Question rewriter
    llm_rewriter = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    system_rewrite = """You a question re-writer that converts an input question to a better version that is optimized
    for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", system_rewrite),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ])
    question_rewriter = re_write_prompt | llm_rewriter | StrOutputParser()

    # Web search
    web_search_tool = TavilySearchResults(k=3)

    return retrieval_grader, rag_chain, question_rewriter, web_search_tool

# Graph functions
def retrieve(state):
    question = state["question"]
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question}

def generate(state):
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            filtered_docs.append(d)
        else:
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def transform_query(state):
    question = state["question"]
    documents = state["documents"]
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

def web_search(state):
    question = state["question"]
    documents = state["documents"]
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents, "question": question}

def decide_to_generate(state):
    web_search = state["web_search"]
    return "transform_query" if web_search == "Yes" else "generate"

# Initialize workflow
def init_workflow():
    class GraphState(Dict):
        question: str
        generation: str
        web_search: str
        documents: List[str]

    workflow = StateGraph(GraphState)

    # Define nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search_node", web_search)

    # Build graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "web_search_node")
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

# Initialize components at startup
print("Initializing CRAG components...")
retriever, vectorstore = init_retriever()
retrieval_grader, rag_chain, question_rewriter, web_search_tool = init_components()
crag_app = init_workflow()
print("CRAG initialization complete!")

# API Models
class Question(BaseModel):
    question: str

class Response(BaseModel):
    answer: str

class UpdateUrlsRequest(BaseModel):
    urls: List[str]

@app.post("/ask", response_model=Response)
async def ask_question(question: Question):
    try:
        result = crag_app.invoke({"question": question.question})
        return Response(answer=result["generation"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_urls")
async def update_urls(request: UpdateUrlsRequest):
    global urls, retriever, vectorstore
    try:
        urls = request.urls
        # Clear the Pinecone DB
        clear_vectorstore(vectorstore)
        # Re-initialize retriever and vectorstore with new URLs
        retriever, vectorstore = init_retriever()
        return {"status": "success", "message": "URLs updated and vectorstore refreshed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
