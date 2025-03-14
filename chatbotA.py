import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain, LLMChain
from langchain.prompts import PromptTemplate
import os
import google.generativeai as genai
import traceback

# Set page configuration
st.set_page_config(page_title="Gemini Chatbot", page_icon="💬")

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationBufferMemory(return_messages=True)
if "chat_active" not in st.session_state:
    st.session_state.chat_active = False
if "summary_displayed" not in st.session_state:
    st.session_state.summary_displayed = False

# App title
st.title("💬 Context-Aware Chatbot")
st.subheader("Powered by Gemini and OpenAI")

# API key input section
with st.sidebar:
    st.header("API Configuration")
    gemini_api_key = st.text_input("Enter your Gemini API Key:", type="password")
    openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    
    # Preferred models list (in order of preference)
    preferred_models = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest"
    ]
    
    # Model selection dropdown
    selected_model = st.selectbox(
        "Select Gemini Model:",
        preferred_models,
        index=0
    )
    
    # Option to use same model for summary
    use_same_model_for_summary = st.checkbox("Use Gemini for summary (if OpenAI quota exceeded)", value=False)
    
    if st.button("Start Chat Session"):
        if gemini_api_key:
            os.environ["GOOGLE_API_KEY"] = gemini_api_key
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key
            
            # Configure Gemini API
            genai.configure(api_key=gemini_api_key)
            
            # Store selected model
            st.session_state.gemini_model = selected_model
            st.session_state.use_same_model_for_summary = use_same_model_for_summary
            st.session_state.chat_active = True
            
            success_message = f"API keys set! Using Gemini model: {selected_model}"
            if not openai_api_key:
                success_message += " (OpenAI key not provided, using Gemini for summary)"
            st.success(success_message)
        else:
            st.error("Please provide at least the Gemini API key.")

# Function to generate summary and sentiment using Gemini
def generate_with_gemini(conversation_text):
    try:
        gemini_llm = ChatGoogleGenerativeAI(
            model=st.session_state.gemini_model,
            google_api_key=gemini_api_key,
            temperature=0.3
        )
        
        summary_prompt = PromptTemplate(
            input_variables=["input"],
            template="""
            Provide a concise summary of this conversation in under 150 words:
            
            {input}
            """
        )
        
        sentiment_prompt = PromptTemplate(
            input_variables=["input"],
            template="""
            Perform a brief sentiment analysis of this conversation. 
            Identify the overall mood (positive, negative, neutral, mixed) and 
            note any significant emotional shifts or patterns.
            Keep your analysis under 50 words and be specific.
            
            {input}
            """
        )
        
        summary_chain = LLMChain(
            llm=gemini_llm,
            prompt=summary_prompt,
            verbose=False
        )
        
        sentiment_chain = LLMChain(
            llm=gemini_llm,
            prompt=sentiment_prompt,
            verbose=False
        )
        
        summary = summary_chain.run(conversation_text)
        sentiment = sentiment_chain.run(conversation_text)
        
        return summary, sentiment
    except Exception as e:
        st.error(f"Error with Gemini summary generation: {str(e)}")
        return f"Error generating summary with Gemini: {str(e)}", "Unable to analyze sentiment due to an error."

# End chat button - More prominently displayed in the main area
if st.session_state.chat_active:
    end_col1, end_col2 = st.columns([5, 1])
    with end_col2:
        end_button = st.button("End Chat", type="primary", use_container_width=True)
        if end_button:
            st.session_state.chat_active = False
            
            # Create the full conversation text
            full_conversation = ""
            for message in st.session_state.messages:
                role = "User" if message["role"] == "user" else "Assistant"
                full_conversation += f"{role}: {message['content']}\n\n"
            
            # Generate summary and sentiment analysis
            with st.spinner("Generating summary and sentiment analysis..."):
                try:
                    # Try OpenAI first if API key is provided and we're not set to use Gemini
                    if openai_api_key and not st.session_state.use_same_model_for_summary:
                        openai_llm = ChatOpenAI(
                            model_name="gpt-3.5-turbo",
                            openai_api_key=openai_api_key,
                            temperature=0.3
                        )
                        
                        summary_prompt = PromptTemplate(
                            input_variables=["input"],
                            template="""
                            Provide a concise summary of this conversation in under 150 words:
                            
                            {input}
                            """
                        )
                        
                        sentiment_prompt = PromptTemplate(
                            input_variables=["input"],
                            template="""
                            Perform a brief sentiment analysis of this conversation. 
                            Identify the overall mood (positive, negative, neutral, mixed) and 
                            note any significant emotional shifts or patterns.
                            Keep your analysis under 50 words and be specific.
                            
                            {input}
                            """
                        )
                        
                        summary_chain = LLMChain(
                            llm=openai_llm,
                            prompt=summary_prompt,
                            verbose=False
                        )
                        
                        sentiment_chain = LLMChain(
                            llm=openai_llm,
                            prompt=sentiment_prompt,
                            verbose=False
                        )
                        
                        summary = summary_chain.run(full_conversation)
                        sentiment = sentiment_chain.run(full_conversation)
                    else:
                        # Use Gemini if OpenAI key not provided or if user chose to use Gemini
                        summary, sentiment = generate_with_gemini(full_conversation)
                        
                except Exception as e:
                    st.warning(f"OpenAI API error: {str(e)}. Falling back to Gemini for summary...")
                    # Fall back to Gemini if OpenAI fails
                    summary, sentiment = generate_with_gemini(full_conversation)
                
                st.session_state.summary = summary
                st.session_state.sentiment = sentiment
                st.session_state.summary_displayed = True
                
            # Reset for new session
            st.session_state.conversation_memory = ConversationBufferMemory(return_messages=True)
            st.rerun()

# Main chat interface
if st.session_state.chat_active:
    # Initialize Gemini chat model
    try:
        llm = ChatGoogleGenerativeAI(
            model=st.session_state.gemini_model,
            google_api_key=gemini_api_key,
            temperature=0.7
        )
        
        # Create conversation chain with memory
        conversation = ConversationChain(
            llm=llm, 
            memory=st.session_state.conversation_memory,
            verbose=False
        )
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("What would you like to talk about?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
                
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = conversation.predict(input=prompt)
                    st.write(response)
            
            # Add AI response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
    except Exception as e:
        st.error(f"Error with Gemini API: {str(e)}")
        st.session_state.chat_active = False
        
elif st.session_state.summary_displayed:
    # Display summary and sentiment analysis when chat session ends
    st.header("Chat Session Summary")
    st.write(st.session_state.summary)
    
    st.header("Sentiment Analysis")
    # Display sentiment with appropriate styling based on content
    sentiment_text = st.session_state.sentiment
    
    # Simple sentiment highlighting based on keywords
    if any(word in sentiment_text.lower() for word in ["positive", "happy", "pleased", "satisfaction"]):
        st.success(sentiment_text)
    elif any(word in sentiment_text.lower() for word in ["negative", "frustrated", "angry", "dissatisfied"]):
        st.error(sentiment_text)
    elif any(word in sentiment_text.lower() for word in ["neutral", "balanced", "mixed"]):
        st.info(sentiment_text)
    else:
        st.write(sentiment_text)
    
    if st.button("Start New Session", type="primary"):
        st.session_state.messages = []
        st.session_state.summary_displayed = False
        st.rerun()
        
else:
    # Instructions when not chatting
    st.info("Please enter your API keys in the sidebar and click 'Start Chat Session' to begin.")