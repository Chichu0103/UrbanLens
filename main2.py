# Install necessary packages
# Note: In a script, it's better to install packages outside of the code.
# The following lines are intended for Jupyter notebooks. If running as a script, ensure packages are installed beforehand.
# !pip install "ibm-watsonx-ai" pydantic>=1.10.0 langchain==0.1.8 langchain_ibm==0.0.1

# imports
import os
import logging
from typing import Dict, Any, List
from prompt import all_prompts

from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes, DecodingMethods

from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from langchain.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Removed unnecessary import for UnstructuredURLLoader since we're not loading from URLs
# from langchain_community.document_loaders import UnstructuredURLLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from duckduckgo_api_haystack import DuckduckgoApiWebSearch

from langchain import PromptTemplate
from langchain.chains import SequentialChain, LLMChain
from langchain.schema import Document  # Ensure LangChain's Document is imported

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_websearch(top_k: int = 10) -> DuckduckgoApiWebSearch:
    """Initialize the DuckDuckGo web search with specified top_k results."""
    return DuckduckgoApiWebSearch(top_k=top_k)

def fetch_search_results(websearch: DuckduckgoApiWebSearch, query: str) -> Dict[str, Any]:
    """Perform web search and retrieve documents and links."""
    results = websearch.run(query=query)
    documents = results.get("documents", [])
    links = results.get("links", [])
    logger.info(f"Fetched {len(documents)} documents and {len(links)} links for query: '{query}'.")
    return {"documents": documents, "links": links}

# Removed create_url_dictionary and load_documents functions as they are no longer needed

from langchain.schema import Document  # Ensure this import is present

def preprocess_documents(documents: List[Document]) -> List[Document]:
    """Preprocess documents by cleaning whitespace and adding metadata."""
    processed_docs = []
    for doc_id, doc in enumerate(documents):
        # Access the text content directly
        cleaned_text = " ".join(doc.page_content.split())  # Remove excessive whitespace

        # Copy existing metadata and add an 'id'
        metadata = dict(doc.metadata) if doc.metadata else {}
        metadata["id"] = doc_id

        # Create a new Document with cleaned text and updated metadata
        processed_doc = Document(page_content=cleaned_text, metadata=metadata)
        processed_docs.append(processed_doc)

        logger.debug(f"Processed document ID: {doc_id}")
    return processed_docs

def split_text(documents: List[Document], chunk_size: int = 512, chunk_overlap: int = 0) -> List[Document]:
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def initialize_embeddings(credentials: Dict[str, str], project_id: str) -> WatsonxEmbeddings:
    """Initialize WatsonxEmbeddings with provided credentials and project ID."""
    return WatsonxEmbeddings(
        model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,
        url=credentials["url"],
        apikey=credentials["apikey"],
        project_id=project_id,
    )

def create_vectorstore(docs: List[Document], embeddings: WatsonxEmbeddings) -> Chroma:
    """Create a Chroma vector store from documents and embeddings."""
    return Chroma.from_documents(documents=docs, embedding=embeddings)

def initialize_llm(credentials: Dict[str, str], project_id: str, model_id: str, parameters: Dict[str, Any]) -> WatsonxLLM:
    """Initialize WatsonxLLM with provided credentials, project ID, model ID, and parameters."""
    return WatsonxLLM(
        model_id=model_id,
        url=credentials.get("url"),
        apikey=credentials.get("apikey"),
        project_id=project_id,
        params=parameters
    )

def build_retrieval_chain(retriever, llm) -> Any:
    """Construct the LangChain processing chain."""
    template = """Answer the question based only on the following context:

{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join([d.page_content for d in docs])

    return (
        {"context": RunnablePassthrough(format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def build_report_chain(llm: WatsonxLLM) -> SequentialChain:
    """Build the LangChain report generation chain."""
    prompt_template = PromptTemplate(
        input_variables=["Input"],
        template="Rephrase and generate a detailed report, with correct punctuation and full sentences. Don't skip any numbers. Use the information provided below. This report should convey all the information in detail.\n\nInformation:\n{Input}\n\nReport:"
    )

    report_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        output_key='Report'
    )

    sequential_chain = SequentialChain(
        chains=[report_chain],
        input_variables=["Input"],
        output_variables=['Report'],
        verbose=True
    )

    return sequential_chain

def process_documents_and_queries(
    queries: List[str],
    credentials: Dict[str, str],
    project_id: str,
    model_id: str,
    top_k_search: int = 3
) -> List[str]:
    """Process multiple queries from web search and retrieve answers using LangChain and Watsonx."""
    # Initialize web search
    websearch = setup_websearch(top_k=top_k_search)

    all_answers = []

    for query in queries:
        logger.info(f"Processing query: {query}")
        search_results = fetch_search_results(websearch, query)
        documents = search_results["documents"]

        if not documents:
            logger.warning(f"No documents found for the query: '{query}'.")
            all_answers.append(f"No information available for query: '{query}'.")
            continue

        # Preprocess and split documents
        documents = preprocess_documents(documents)
        docs = split_text(documents)

        # Initialize embeddings and vector store
        embeddings = initialize_embeddings(credentials, project_id)
        vectorstore = create_vectorstore(docs, embeddings)
        retriever = vectorstore.as_retriever()

        # Define generation parameters
        parameters = {
            GenParams.DECODING_METHOD: DecodingMethods.SAMPLE.value,
            GenParams.MAX_NEW_TOKENS: 100,
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.TEMPERATURE: 0.5,
            GenParams.TOP_K: 50,
            GenParams.TOP_P: 1
        }

        # Initialize LLM
        llm = initialize_llm(credentials, project_id, model_id, parameters)

        # Construct and invoke the chain
        chain = build_retrieval_chain(retriever, llm)
        try:
            response = chain.invoke({"context": docs, "question": query})
            logger.info(f"Query processed successfully: {query}")
            all_answers.append(response)
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            all_answers.append(f"An error occurred while processing the query: '{query}'.")

    return all_answers

def generate_comprehensive_report(answers: List[str], credentials: Dict[str, str], project_id: str, model_id: str, parameters: Dict[str, Any]) -> str:
    """Generate a comprehensive report based on the collected answers."""
    # Initialize LLM for report generation
    report_llm = WatsonxLLM(
        model_id=model_id,
        url=credentials["url"],
        apikey=credentials["apikey"],
        project_id=project_id,
        params=parameters
    )

    # Build the report generation chain
    report_chain = build_report_chain(report_llm)

    # Aggregate answers into a single input string
    aggregated_input = "\n\n".join(answers)

    # Invoke the report generation chain
    try:
        report = report_chain.invoke({"Input": aggregated_input})
        logger.info("Report generated successfully.")
        return report
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return "An error occurred while generating the report."

def main():
    # Configuration
    credentials = {
        "url": "https://us-south.ml.cloud.ibm.com",
        "apikey": "_q6TjVl8frAXoZE34-lfU-KJgvqFhC6KdDqNWzNF-ezR"  # Replace with secure method in production
    }
    project_id = "78719d7a-04be-4e01-ae9a-d7db25e2a936"
    model_id = ModelTypes.GRANITE_13B_CHAT_V2.value

    # Define queries
    prompt = all_prompts['crime']
    print(prompt)

    # Ensure that prompt is a list of queries. If it's a single query string, wrap it in a list.
    if isinstance(prompt, str):
        queries = [prompt]
    elif isinstance(prompt, list):
        queries = prompt
    else:
        logger.error("Invalid format for prompts. Expected a string or list of strings.")
        queries = []

    # Process queries to get answers
    answers = process_documents_and_queries(
        queries=queries,
        credentials=credentials,
        project_id=project_id,
        model_id=model_id,
        top_k_search=10
    )

    # Define parameters for report generation
    report_parameters = {
        GenParams.DECODING_METHOD: DecodingMethods.SAMPLE.value,
        GenParams.MAX_NEW_TOKENS: 500,  # Increased token limit for a more comprehensive report
        GenParams.MIN_NEW_TOKENS: 50,
        GenParams.TEMPERATURE: 0.7,  # Slightly higher temperature for more creative reports
        GenParams.TOP_K: 50,
        GenParams.TOP_P: 0.9
    }

    # Generate the comprehensive report
    report = generate_comprehensive_report(
        answers=answers,
        credentials=credentials,
        project_id=project_id,
        model_id=model_id,
        parameters=report_parameters
    )

    # Print the final report
    print("\n=== Comprehensive Report ===\n")
    print(report)

if __name__ == "__main__":
    main()
