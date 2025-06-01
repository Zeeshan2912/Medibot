from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

DATA_PATH = "data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data,
                              glob = '*.pdf',
                              loader_cls = PyPDFLoader)

    documents = loader.load()
    return documents


documents = load_pdf_files(data=DATA_PATH)
#print("Length of PDF pages: ", len(documents))


def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)
#print("Length of text chunks: ", len(text_chunks))


def get_embedding_model():
    huggingface_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN") # Changed to HUGGINGFACEHUB_API_TOKEN
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        huggingfacehub_api_token=huggingface_api_key # Added API key
    )
    return embedding_model

# Optionally, you can add a test to verify chat completion works here as well:
def ask_llm_chat_completion(user_message):
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    client = InferenceClient(
        provider="hf-inference",
        api_key=token,
    )
    completion = client.chat.completions.create(
        model="Qwen/Qwen3-235B-A22B",
        messages=[
            {
                "role": "user",
                "content": user_message
            }
        ],
    )
    return completion.choices[0].message.content

# Example usage (uncomment to test):
# print(ask_llm_chat_completion("What is the capital of France?"))

# Optionally, you can add a test to verify chat completion works here as well:
def ask_llm_chat_completion_local(user_message):
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    model_id = "tiiuae/falcon-7b-instruct"  # Changed to an open-access model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    result = pipe(user_message, max_new_tokens=50)
    return result[0]['generated_text']

# Example usage (uncomment to test):
# print(ask_llm_chat_completion_local("What is the capital of France?"))

embedding_model = get_embedding_model()

DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(
    text_chunks,
    embedding_model,
)

db.save_local(DB_FAISS_PATH)
