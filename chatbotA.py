import streamlit as st
from openai import OpenAI
import google.generativeai as genai

# Set page configuration
st.set_page_config(page_title="Context-Aware Chatbot", page_icon="🤖", layout="wide")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_keys_entered" not in st.session_state:
    st.session_state.api_keys_entered = False

# Title and description
st.title("Context-Aware Chatbot")
st.subheader("Combining OpenAI and Gemini for enhanced responses")

# API keys input
if not st.session_state.api_keys_entered:
    with st.form("api_keys_form"):
        st.header("Enter API Keys")
        openai_key = st.text_input("OpenAI API Key", type="password")
        gemini_key = st.text_input("Google Gemini API Key", type="password")
        
        submit_button = st.form_submit_button("Submit")
        
        if submit_button:
            if openai_key and gemini_key:
                st.session_state.openai_api_key = openai_key
                st.session_state.gemini_api_key = gemini_key
                st.session_state.api_keys_entered = True
                st.rerun()  # Updated from experimental_rerun to rerun
            else:
                st.error("Both API keys are required.")

# Only show the main app if API keys are provided
if st.session_state.api_keys_entered:
    # Initialize API clients
    openai_client = OpenAI(api_key=st.session_state.openai_api_key)
    genai.configure(api_key=st.session_state.gemini_api_key)
    
    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.info("""This chatbot first generates a response using OpenAI's model, 
                 then refines it with Google's Gemini model to provide more context-rich and accurate answers.""")
        
        # Option to reset API keys
        if st.button("Reset API Keys"):
            st.session_state.api_keys_entered = False
            st.rerun()  # Updated from experimental_rerun to rerun

    # Function to get response from OpenAI
    def get_openai_response(prompt, context=""):
        try:
            full_prompt = f"{context}\n\nUser question: {prompt}"
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error with OpenAI: {str(e)}"

    # Function to refine with Gemini
    def refine_with_gemini(original_prompt, openai_response):
        try:
            # Create a Gemini model instance
            gemini_model = genai.GenerativeModel(model_name="gemini-1.5-pro")
            
            refinement_prompt = f"""
            Original user question: {original_prompt}
            
            Initial response: {openai_response}
            
            Please refine and enhance the above response. Add more context, correct any inaccuracies, 
            and make it more comprehensive while keeping it concise. If the initial response is already excellent,
            you can return it with minimal changes.
            """
            
            response = gemini_model.generate_content(refinement_prompt)
            return response.text
        except Exception as e:
            return f"Error with Gemini refinement: {str(e)}\nOriginal response: {openai_response}"

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Display model sources if available
            if "source" in message:
                st.caption(f"Source: {message['source']}")

    # Chat input
    if prompt := st.chat_input("Ask something..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get context from previous messages (last 10 exchanges)
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-10:]])
        
        # Show a spinner while processing
        with st.spinner("Thinking..."):
            # Get OpenAI response
            with st.chat_message("assistant"):
                openai_response_container = st.empty()
                openai_response = get_openai_response(prompt, context)
                openai_response_container.write(openai_response)
                st.caption("Initial response from OpenAI")
                
                # Refine with Gemini
                refined_response = refine_with_gemini(prompt, openai_response)
                st.write("---")
                st.write(refined_response)
                st.caption("Refined by Gemini")
        
        # Add final response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": refined_response,
            "source": "OpenAI + Gemini"
        })

    # Add explanatory expander
    with st.expander("How this works"):
        st.write("""
        1. Your question is sent to OpenAI's GPT model to generate an initial response
        2. The initial response is then sent to Google's Gemini model
        3. Gemini analyzes the original question and the initial response
        4. Gemini refines the response by adding context, correcting inaccuracies, and enhancing details
        5. The refined response is presented to you
        
        This dual-model approach leverages the strengths of both AI systems to provide more comprehensive and accurate answers.
        """)