import os
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.document_loaders import OutlookMessageLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from datetime import datetime
from langchain.globals import set_verbose
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = "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"

# Set verbose to True for detailed logs
set_verbose(True)

# Load environment variables
load_dotenv()

# Set API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create an instance of ChatOpenAI with the specified model
llm = ChatOpenAI(model="gpt-4o-mini")

# Initialize OpenAI Embeddings
class CustomOpenAIEmbeddings(OpenAIEmbeddings):
    def embed_documents(self, docs):
        return super().embed_documents(docs)
    
    def embed_query(self, query):
        return super().embed_query(query)

embedding = CustomOpenAIEmbeddings()

# Streamlit UI
st.title("Email Summarizer and Chat with GPT-4o")

# Determine the absolute path to the 'data/email' directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_folder_path = os.path.join(script_dir, "data", "output")
persist_directory = 'docs/chroma/v1'

# List to hold all documents
all_documents = []

# Load emails and process only if embeddings are not already cached
if os.path.exists(persist_directory) and os.listdir(persist_directory):
    st.write("Loading embeddings from cache...")
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
else:
    # Iterate through each subfolder in the parent directory
    if os.path.exists(parent_folder_path):
        for root, dirs, files in os.walk(parent_folder_path):
            print("Files in directory:", files)
            for filename in files:
                if filename.lower().endswith(".msg"):
                    msg_file_path = os.path.join(root, filename)
                    loader = OutlookMessageLoader(msg_file_path)
                    email_documents = loader.load()
                    for doc in email_documents:
                        doc.metadata["source"] = os.path.basename(loader.file_path)
                        # Convert datetime metadata to string
                        for key, value in doc.metadata.items():
                            if isinstance(value, datetime):
                                doc.metadata[key] = value.isoformat()
                        all_documents.append(doc)
        
        print("Total documents loaded:", len(all_documents))

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150
        )
        
        splits = text_splitter.split_documents(all_documents)
        print("Total document splits:", len(splits))
        
        # Debugging: Print the number of splits
        st.write(f"Number of document splits: {len(splits)}")

        st.write("Creating new embeddings...")
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embedding,  # Ensure only one embedding argument is passed
            persist_directory=persist_directory
        )
    
    # Debugging: Print the number of documents added to the vector store
    st.write(f"Number of documents in the vector store: {vectordb._collection.count()}")

# Initialize conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Check if there are any document splits
if vectordb._collection.count() > 0:
    st.subheader("Chat with GPT-4o about the emails")
    st.write()
    # Chat functionality
    user_input = st.text_input("You: ", key="user_input")
    
    if st.button("Send"):
        if user_input:
            # Update conversation history
            st.session_state.conversation_history.append(f"You: {user_input}")
            
            # Perform similarity search to find relevant documents
            search_results = vectordb.similarity_search(user_input, k=2)
            
            # Build context from conversation history
            context = "\n".join(st.session_state.conversation_history[-5:])  # Limit to the last 5 exchanges

            # Build prompt
            template = """Use the following pieces of context to answer the question at the end. If you don't 
            know the answer, just say that you don't know, don't try to make up an answer. Use three sentences 
            maximum. Keep the answer as concise as possible. Always say "thanks for asking the PC AI BOT!" at 
            the end of the answer. 
            {context}
            Question: {question}
            Helpful Answer:"""
            QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
                
            # Run chain
            qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=vectordb.as_retriever(),
                return_source_documents=True,
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
            )
            chat_response = qa_chain({"query": user_input})
            
            # Extract the formatted response
            formatted_response = chat_response["result"]

            # Update conversation history with the bot's response
            st.session_state.conversation_history.append(f"Bot: {formatted_response}")

            # Display the OpenAI response
            st.text_area("ChatGPT:", value=formatted_response, height=200, max_chars=None, key=None)
            
            # Display the message content from the similarity search results
            st.subheader("Relevant Documents")
            for result in search_results:
                st.write(f"Date: {result.metadata['date']}")
                st.write(f"Source: {result.metadata['source']}")
                st.write(f"Content: {result.page_content}")

            st.subheader("Conversation History")
            for exchange in st.session_state.conversation_history:
                st.write(exchange)
else:
    st.error(f"Folder not found: {parent_folder_path}")

# Debugging: Print documents for verification
# for doc in all_documents:
#     st.write(f"Document from {doc.metadata['source']}:")
#     st.write(doc.page_content)
