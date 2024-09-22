# Install necessary packages
# Note: In a script, it's better to install packages outside of the code.
# The following lines are intended for Jupyter notebooks. If running as a script, ensure packages are installed beforehand.
# !pip install "ibm-watsonx-ai" pydantic>=1.10.0 langchain==0.1.8 langchain_ibm==0.0.1

# imports
import os
from prompt import generate_prompts
import logging
from typing import Dict, Any, List
from prompt import generate_prompts

from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes, DecodingMethods

from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.document_loaders import UnstructuredURLLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from duckduckgo_api_haystack import DuckduckgoApiWebSearch

from langchain import PromptTemplate
from langchain.chains import SequentialChain, LLMChain

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
    logger.info(f"Fetched {len(links)} links for query: '{query}'.")
    return {"documents": documents, "links": links}

def create_url_dictionary(links: list) -> Dict[str, str]:
    """Create a dictionary mapping unique IDs to URLs."""
    return {f"link_id_{i + 1}": link for i, link in enumerate(links)}

def load_documents(urls: list) -> list:
    """Load documents from a list of URLs."""
    documents = []
    for url in urls:
        try:
            loader = UnstructuredURLLoader(urls=[url])
            data = loader.load()
            documents.extend(data)
            logger.info(f"Loaded data from {url}")
        except Exception as e:
            logger.error(f"Error loading {url}: {e}")
    return documents

def preprocess_documents(documents: list) -> list:
    """Preprocess documents by cleaning whitespace and adding metadata."""
    for doc_id, doc in enumerate(documents):
        doc.page_content = " ".join(doc.page_content.split())  # Remove excessive whitespace
        doc.metadata["id"] = doc_id
        logger.debug(f"Processed document ID: {doc_id}")
    return documents

def split_text(documents: list, chunk_size: int = 400, chunk_overlap: int = 0) -> list:
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

def create_vectorstore(docs: list, embeddings: WatsonxEmbeddings) -> Chroma:
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

    def format_docs(docs):
        # Combine documents into a single string
        combined_content = "\n\n".join([d.page_content for d in docs])
        
        # Check if combined content exceeds a safe token limit (e.g., 500)
        if len(combined_content.split()) > 400:  # Adjust based on your token counting method
            # Trim the content
            combined_content = " ".join(combined_content.split()[:400])  # Keep the first 500 words
            logger.warning("Context exceeded token limit and was trimmed.")

        return combined_content


    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def build_report_chain(llm: WatsonxLLM) -> SequentialChain:
    """Build the LangChain report generation chain."""
    prompt_template = PromptTemplate(
        input_variables=["Input"],
        template="rephrase and present The information  along with numbers in more readble way .\n\nInformation:\n{Input}\n\nReport:"
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
        links = search_results["links"]

        if not links:
            logger.warning(f"No links found for the query: '{query}'.")
            all_answers.append(f"No information available for query: '{query}'.")
            continue

        # Create URL dictionary and load documents
        url_dict = create_url_dictionary(links)
        documents = load_documents(list(url_dict.values()))

        if not documents:
            logger.warning(f"No documents loaded from the provided links for query: '{query}'.")
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
            GenParams.TEMPERATURE: 0.6,
            GenParams.TOP_K: 50,
            GenParams.TOP_P: 1
        }

        # Initialize LLM
        llm = initialize_llm(credentials, project_id, model_id, parameters)

        # Construct and invoke the chain
        chain = build_retrieval_chain(retriever, llm)
        try:
            response = chain.invoke(query)
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
        model_id=ModelTypes.GRANITE_13B_CHAT_V2.value,
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

def get_output(prompt,location,top_k_s):
    all_prompts=generate_prompts(location)
    # Configuration
    
        
    credentials = {
            "url": "https://us-south.ml.cloud.ibm.com",
            "apikey": "_q6TjVl8frAXoZE34-lfU-KJgvqFhC6KdDqNWzNF-ezR"  # Replace with secure method in production
        }
    project_id = "78719d7a-04be-4e01-ae9a-d7db25e2a936"
    model_id = ModelTypes.GRANITE_13B_CHAT_V2.value

    

    # Process queries to get answers
    answers = process_documents_and_queries(
        queries=all_prompts[prompt],
        credentials=credentials,
        project_id=project_id,
        model_id=model_id,
        top_k_search=top_k_s
    )   

    
    report_parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.SAMPLE.value,
    GenParams.MAX_NEW_TOKENS: 200,
    GenParams.MIN_NEW_TOKENS: 100,
    GenParams.TEMPERATURE: 0.1,
    GenParams.TOP_K: 50,
    GenParams.TOP_P: 1

    }

  
    report = generate_comprehensive_report(
        answers=answers,
        credentials=credentials,
        project_id=project_id,
        model_id=model_id,
        parameters=report_parameters
    )

    
    return report['Report']





# Define credentials and other necessary parameters

