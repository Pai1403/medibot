import streamlit as st
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory

# 🔹 Initialize LangChain's Ollama LLM
llm = Ollama(base_url='http://localhost:11434', model='mistral-nemo')

# 🔹 Set up Streamlit UI
st.set_page_config(page_title="🩺 Medical Chatbot", layout="wide")
st.title("🩺 Medical Chatbot with Memory 🤖")

# 🔹 Initialize memory in session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

# 🔹 Strong medical chatbot prompt
def construct_prompt(user_query):
    past_conversation = st.session_state.memory.load_memory_variables({}).get("history", "")
    prompt = f"""
    You are a highly knowledgeable **medical AI assistant**.
    Your responses must be **clear, concise, and based on trusted medical sources** like WHO, Mayo Clinic, and PubMed.
    
    - If the user asks a medical question, provide **accurate, evidence-based answers**.
    - If unsure, say: *"I recommend consulting a healthcare professional."*
    - Keep responses **short but informative** (3-4 sentences max).
    - Use **layman-friendly language**, unless the user requests detailed medical terminology.

    --- Chat History ---
    {past_conversation}

    User: {user_query}
    AI:
    """
    return prompt

# 🔹 User input
user_input = st.text_input("👤 Ask a medical question:")

if st.button("Send"):
    if user_input:
        # Construct prompt
        final_prompt = construct_prompt(user_input)

        # Generate response using Mistral-Nemo
        bot_response = llm.invoke(final_prompt)

        # Store in memory
        st.session_state.memory.save_context({"user": user_input}, {"ai": bot_response})

        # Display response
        st.text_area("🤖 AI:", value=bot_response, height=200)

# 🔹 Show chat history
st.subheader("📝 Chat History")
history = st.session_state.memory.load_memory_variables({}).get("history", "")
st.text_area("Chat Memory", value=history, height=200)
