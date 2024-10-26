import streamlit as st
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Load the BlenderBot model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Initialize Streamlit app
st.title("AI Chatbot")
st.write("Ask me anything! Type 'exit' to end the conversation.")

# Conversation history to store user inputs and bot responses
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("You:", "")

# Chatbot response when there's user input
if user_input:
    # Exit condition
    if user_input.lower() == 'exit':
        st.write("Chatbot: Goodbye!")
    else:
        # Tokenize and get response from model
        inputs = tokenizer(user_input, return_tensors="pt")
        response_ids = model.generate(inputs['input_ids'], max_length=100, pad_token_id=tokenizer.eos_token_id)
        bot_response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        
        # Display conversation
        st.session_state.chat_history.append((user_input, bot_response))

        # Show chat history
        for user_msg, bot_msg in st.session_state.chat_history:
            st.write(f"You: {user_msg}")
            st.write(f"Chatbot: {bot_msg}")
        
        # Clear the input box after sending
        st.experimental_rerun()
