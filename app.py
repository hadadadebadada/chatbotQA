import os
import streamlit as st
from langchain import OpenAI, ConversationChain

os.environ['OPENAI_API_KEY'] = "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"

st.title('OR-Chatbot')

llm = OpenAI(temperature=0.9)
conversation = ConversationChain(llm=llm, verbose=True)

# Check if the conversation chain exists in session state, if not, create it
if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = conversation

# Initialize the list of previous responses in session state
if 'previous_responses' not in st.session_state:
    st.session_state.previous_responses = []

# Initialize the list of previous prompts in session state
if 'previous_prompts' not in st.session_state:
    st.session_state.previous_prompts = []

prompt = st.text_input("Stelle deine Frage!")

if prompt:
    response = st.session_state.conversation_chain.predict(input=prompt)

    # Add the prompt and response to the list of previous prompts and responses
    st.session_state.previous_prompts.append(prompt)
    st.session_state.previous_responses.append(response)
    

# Display the list of previous prompts and responses in reverse order
if st.session_state.previous_prompts and st.session_state.previous_responses:
    #st.write("Vorherige Unterhaltungen")
    for i in range(len(st.session_state.previous_prompts)-1, -1, -1):  # Reverse loop
        color = "#606060" if i % 2 == 0 else "#808080"
        st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 5px;'>"
                    f"<b>Q:</b> {st.session_state.previous_prompts[i]}<br>"
                    f"<b>A:</b> {st.session_state.previous_responses[i]}"
                    f"</div>", unsafe_allow_html=True)

