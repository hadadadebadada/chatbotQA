import os
import streamlit as st
from langchain import OpenAI, ConversationChain
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA


from langchain.chat_models import ChatOpenAI

os.environ['OPENAI_API_KEY'] = "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"

# Set up to use existing DB
persist_directory = 'db'
embedding = OpenAIEmbeddings()

vectordb2 = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding)



retriever = vectordb2.as_retriever(search_kwargs={"k": 2})

# Set up the turbo LLM
turbo_llm = ChatOpenAI(
    temperature=0,
    model_name='gpt-3.5-turbo'
)

# Create the chain to answer questions 
qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm, 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)

# Function to process and display the LLM's responses
def process_llm_response(llm_response):
    response_text = llm_response['result']
    sources = '\n'.join([source.metadata['source'] for source in llm_response["source_documents"]])
    return response_text + "\n\nSources:\n\n" + sources

st.title('OR-Chatbot')

# Initialize the list of previous responses in session state
if 'previous_responses' not in st.session_state:
    st.session_state.previous_responses = []

# Initialize the list of previous prompts in session state
if 'previous_prompts' not in st.session_state:
    st.session_state.previous_prompts = []

prompt = st.text_input("Stelle deine Frage!")

if prompt:
    llm_response = qa_chain(prompt)
    response = process_llm_response(llm_response)

    # Add the prompt and response to the list of previous prompts and responses
    st.session_state.previous_prompts.append(prompt)
    st.session_state.previous_responses.append(response)

# Display the list of previous prompts and responses in reverse order
if st.session_state.previous_prompts and st.session_state.previous_responses:
    for i in range(len(st.session_state.previous_prompts)-1, -1, -1):  # Reverse loop
        color = "#606060" if i % 2 == 0 else "#808080"
        st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 5px;'>"
                    f"<b>Q:</b> {st.session_state.previous_prompts[i]}<br>"
                    f"<b>A:</b> {st.session_state.previous_responses[i]}"
                    f"</div>", unsafe_allow_html=True)