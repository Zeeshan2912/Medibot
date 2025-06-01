import os
os.environ['STREAMLIT_WATCHER_TYPE'] = 'none'  # Disable Streamlit watcher at early stage
import sys
import types
try:
    import torch
    # Create a dummy module for torch._classes to avoid inspection errors
    dummy_mod = types.ModuleType('torch._classes')
    dummy_mod.__path__ = []
    sys.modules['torch._classes'] = dummy_mod
except ImportError:
    pass

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from transformers import pipeline

# Define CUSTOM_PROMPT_TEMPLATE globally
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

@st.cache_resource(show_spinner=False)
def get_db():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    DB_FAISS_PATH = "vectorstore/db_faiss"
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# @st.cache_resource(show_spinner=False) # Removed caching for LLM as per previous user request
def get_llm():
    # Using a smaller model for potentially faster local inference on CPU
    model_id = "EleutherAI/gpt-neo-1.3B" 
    
    # Use pipeline directly instead of initializing model and tokenizer separately
    # This lets the pipeline handle the loading using the version-appropriate methods
    from langchain_huggingface import HuggingFacePipeline
    
    pipe = pipeline(
        "text-generation", 
        model=model_id,  # Pass the model_id directly
        max_new_tokens=256, # Adjusted for potentially faster responses
        pad_token_id=50256  # Set pad_token_id here
    )
    hf_llm = HuggingFacePipeline(pipeline=pipe)
    return hf_llm

db = get_db()
llm = get_llm()

def set_custom_prompt(custom_prompt_template_str): # Renamed arg to avoid conflict
    prompt = PromptTemplate(template=custom_prompt_template_str, input_variables=["context", "question"])
    return prompt

def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    user_prompt = st.chat_input("Pass your prompt here") # Renamed to avoid conflict with prompt variable from set_custom_prompt

    if user_prompt:
        if not user_prompt.strip():
            st.warning("Please enter a question.")
            st.session_state.messages.append({"role": "assistant", "content": "Please enter a question."})
        else:
            st.chat_message('user').markdown(user_prompt)
            st.session_state.messages.append({"role": "user", "content": user_prompt})

            try:
                if db is None:
                    st.error("Failed to load the vector store")
                    return # Exit if db is not loaded

                if llm is None:
                    st.error("Failed to load the LLM")
                    return # Exit if llm is not loaded

                # Use the global CUSTOM_PROMPT_TEMPLATE
                qa_prompt = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=db.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': qa_prompt}
                )

                response = qa_chain.invoke({'query': user_prompt})

                result = response["result"].strip()
                source_documents = response["source_documents"]
                # If the model failed to generate a meaningful answer (e.g., echoed the prompt), extract fallback answer
                def extract_answer_from_docs(docs):
                    if docs:
                        text = docs[0].page_content
                        # Split into sentences
                        import re
                        sentences = re.split(r'(?<=[.!?])\s+', text)
                        if sentences:
                            return sentences[0].strip()
                    return "I don't know."
                # Detect placeholder echo and fallback
                if not result or result.startswith("Use the pieces"):
                    result = extract_answer_from_docs(source_documents)

                # Format source documents for better readability
                formatted_source_docs = "\n\nSource Documents:\n"
                for i, doc in enumerate(source_documents):
                    formatted_source_docs += f"Doc {i+1}: {doc.page_content[:200]}...\n" # Display first 200 chars
                    if hasattr(doc, 'metadata') and doc.metadata:
                        formatted_source_docs += f"  Source: {doc.metadata.get('source', 'N/A')}\n"


                result_to_show = result + formatted_source_docs
                
                st.chat_message('assistant').markdown(result_to_show)
                st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

            except Exception as e:
                st.error(f"Error during RAG pipeline: {str(e)}")
                # Optionally, log the full traceback for debugging
                # import traceback
                # st.error(traceback.format_exc())


if __name__ == "__main__":
    main()