from model import *
import streamlit as st


@st.cache_resource()
def my_model():
    qa_chain = setup_and_initialize()
    return qa_chain
st.title("Game of Thrones GPT")
hi = "hi"
if "custom_model" not in st.session_state:
    st.session_state["custom_model"] = my_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        input_text = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
        print(input_text[-1])
        qa_chain = st.session_state["custom_model"]
        output_text = qa_chain(input_text[-1]) #YOUR_MODEL_PREDICTION_FUNCTION(input_text)

        print(output_text['answer'])
        message_placeholder.markdown(output_text['answer'])
    st.session_state.messages.append({"role": "assistant", "content": output_text['answer']})
