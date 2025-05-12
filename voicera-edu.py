import streamlit as st
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Simple Chatbot",
    layout="centered"
)

# Custom CSS for chat bubbles
st.markdown("""
<style>
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 8px;
        margin-bottom: 20px;
    }
    .chat-bubble {
        max-width: 80%;
        padding: 12px 16px;
        border-radius: 16px;
        margin-bottom: 4px;
        font-size: 14px;
        line-height: 1.4;
    }
    .user-bubble {
        background-color: #f0f0f0;
        align-self: flex-end;
        border-bottom-right-radius: 4px;
    }
    .bot-bubble {
        background-color: #e3f2fd;
        align-self: flex-start;
        border-bottom-left-radius: 4px;
    }
    .support-header {
        background-color: #e3f2fd;
        padding: 8px 16px;
        border-radius: 16px;
        font-weight: bold;
        margin-bottom: 8px;
    }
    .timestamp {
        font-size: 11px;
        color: #666;
        margin-top: 4px;
    }
    .message-input {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        width: 80%;
        max-width: 600px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi. I'm here to help you :)"},
        {"role": "assistant", "content": "Hello, thanks for visiting"}
    ]

# Display chat messages
st.title("CHATBOT")
st.markdown('<div class="support-header">Support</div>', unsafe_allow_html=True)

chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        bubble_class = "user-bubble" if message["role"] == "user" else "bot-bubble"
        timestamp = datetime.now().strftime("%H:%M")
        st.markdown(f"""
        <div class="chat-bubble {bubble_class}">
            {message["content"]}
            <div class="timestamp">{timestamp}</div>
        </div>
        """, unsafe_allow_html=True)

# User input
input_container = st.container()
with input_container:
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Write a message...", key="input", label_visibility="collapsed")
        submitted = st.form_submit_button("Send")

# Handle user input
if submitted and user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate bot response (simple hardcoded responses for demo)
    if "what do you sell" in user_input.lower():
        bot_response = "We sell coffee and tea"
    elif "pay with paypal" in user_input.lower():
        bot_response = "We accept most major credit cards, and Paypal"
    elif "github.com" in user_input.lower():
        bot_response = "You can visit our GitHub page at github.com/yourusername"
    else:
        bot_response = "I'm sorry, I didn't understand that. How can I help you?"
    
    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    
    # Rerun to update the chat display
    st.rerun()