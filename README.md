**🩺 MediBot: Medical Chatbot with Memory**

MediBot is a medical assistant chatbot built using the Mistral 7B model served via Ollama. It provides evidence-based answers to medical queries, ensuring accurate, reliable, and concise responses. The chatbot stores conversation history and leverages LangChain for memory management.

**🚀 Features**
- Local deployment of Mistral-nemo 12B via Ollama.
- Provides clear, concise, and medically accurate responses.
- Memory storage that retains conversation history.
- User-friendly Streamlit interface.

**🔧 Prerequisites**
Before running the project, make sure the following dependencies are installed:
- Python 3.7+
- Ollama (local LLM server running on your machine)
- Git clone the Repo
Bash: 
```pip install -r requirements.txt```

Run the Streamlit App:
Start the Streamlit app by running the following command:

```streamlit run mediBot_prompt.py```

The app should open in your browser, where you can interact with the medical chatbot.

**📝 How to Use**
- Ask Questions: Type a medical question in the input box. The bot will provide accurate, concise, and medically reliable answers.
- Chat History: The chatbot stores previous interactions and includes them in the conversation, ensuring a continuous chat flow.
- Memory Management: The bot remembers previous conversations for context, and you can see the memory stored on the side of the interface.

**🔧 Customizing the Chatbot**
- Modify the Prompt: You can adjust the medical chatbot prompt in the construct_prompt() function to refine the bot's personality or adjust how it responds.
- Switch Models: If you want to try a different model, update the model name in the Ollama() instantiation (for example, model='mistral-nemo').
- Adjust Memory: The chatbot stores previous conversation history using LangChain's ConversationBufferMemory. You can configure the amount of history it keeps by modifying the st.session_state.memory settings.

**🤖 How It Works**
- Local LLM Deployment: The chatbot uses Ollama to serve the Mistral-nemo with model with 128k context length running locally.
- Prompt Design: The chatbot uses a structured medical prompt to ensure answers are both medically accurate and concise.
- Memory Management: It leverages LangChain’s ConversationBufferMemory to store and retrieve previous conversations, allowing the chatbot to understand context.
- Streamlit Interface: Users interact with the bot via a simple, intuitive Streamlit interface.

**🛠️ Future Improvements**
- RAG (Retrieval-Augmented Generation): Integrate ChromaDB for storing and retrieving long-term memory.
- Voice Integration: Add voice input and output for more accessible interaction.
- Multi-turn Conversations: Enhance conversation handling for more complex, multi-turn interactions.
  
📄 License
This project is open source and available under the MIT License. See the LICENSE file for more details.

