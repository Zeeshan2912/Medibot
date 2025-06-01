import  os
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()   


HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN") 
HUGGINGFACE_REPO_ID = "google/flan-t5-small" # Changed model to align with direct API testing

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def ask_llm_chat_completion(user_message):
    # Use local inference to avoid API credit issues
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    model_id = "EleutherAI/gpt-neo-1.3B"  # Changed to a smaller open-access model for faster CPU inference
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    result = pipe(user_message, max_new_tokens=50)
    return result[0]['generated_text']

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

user_query = input("Enter your question: ")
llm_response = ask_llm_chat_completion(user_query)
print("LLM RESPONSE:", llm_response)