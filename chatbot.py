import streamlit as st
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Model setup inside the function to reduce potential issues on Streamlit Cloud
@st.cache_resource
def load_model():
    model_name = "facebook/blenderbot-400M-distill"
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

# Load model and tokenizer
tokenizer, model = load_model()

# Streamlit UI
st.title("AI Chatbot")
st.write("Ask me anything! Type 'exit' to end the conversation.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("You:", "")

# Chatbot response
if user_input:
    # Exit condition
    if user_input.lower() == 'exit':
        st.write("Chatbot: Goodbye!")
    else:
        # Generate response
        inputs = tokenizer(user_input, return_tensors="pt")
        response_ids = model.generate(inputs['input_ids'], max_length=100, pad_token_id=tokenizer.eos_token_id)
        bot_response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        
        # Store and display conversation
        st.session_state.chat_history.append((user_input, bot_response))
        for user_msg, bot_msg in st.session_state.chat_history:
            st.write(f"You: {user_msg}")
            st.write(f"Chatbot: {bot_msg}")
        
        # Clear the input box after sending
        st.experimental_rerun()
