import streamlit as st
from PIL import Image
import google.generativeai as genai
from wsec_helper import WSECComplianceAssistant

# Page configuration
st.set_page_config(
    page_title="Washington State Energy Code Assistant",
    page_icon="⚡",
    layout="wide"
)

# Initialize session state
if 'chat' not in st.session_state:
    st.session_state.chat = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'image_analyzed' not in st.session_state:
    st.session_state.image_analyzed = False
if 'current_image' not in st.session_state:
    st.session_state.current_image = None

# Sidebar for API key
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    if api_key:
        assistant = WSECComplianceAssistant(api_key)
    else:
        assistant = WSECComplianceAssistant()
    
    # Add clear chat button
    if st.button("Clear Analysis History"):
        st.session_state.messages = []
        st.session_state.chat = assistant.start_chat()
        st.session_state.image_analyzed = False
        st.rerun()

# Main title
st.title("⚡ Washington State Energy Code Compliance Assistant")
st.markdown("""
This AI assistant helps architects and engineers verify compliance with the Washington State Energy Code (WSEC). 
Discuss code requirements, upload architectural plans, and receive detailed compliance analysis.
""")

# Initialize chat if not already done
if st.session_state.chat is None:
    st.session_state.chat = assistant.start_chat()

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["💬 WSEC Discussion", "📋 Plan Review"])

with tab1:
    st.markdown("### Energy Code Consultation")
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about Washington State Energy Code requirements..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing WSEC Requirements..."):
                response = assistant.send_message(st.session_state.chat, prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

with tab2:
    st.markdown("### Architectural Plan Review")
    # File uploader
    uploaded_file = st.file_uploader("Upload architectural plans or specification documents", type=['png', 'jpg', 'jpeg', 'pdf'])

    if uploaded_file:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Architectural Plan", use_container_width=True)
        
        # Analyze button
        if not st.session_state.image_analyzed:
            if st.button("Perform WSEC Compliance Review", type="primary"):
                with st.spinner("Conducting detailed WSEC compliance analysis..."):
                    # Store image for reference
                    st.session_state.current_image = image
                    
                    # Get initial analysis
                    report = assistant.analyze_image(image, st.session_state.chat)
                    
                    # Add the report to chat history
                    st.session_state.messages.append({"role": "assistant", "content": report})
                    st.session_state.image_analyzed = True
                    st.rerun()

# Footer with instructions
st.markdown("---")
st.markdown("""
### How to Use the WSEC Compliance Assistant
1. Chat about Energy Code requirements in the Discussion tab
2. For detailed plan reviews:
   - Switch to the Plan Review tab
   - Upload architectural plans or specification documents
   - Click "Perform WSEC Compliance Review" for a comprehensive analysis
3. Receive detailed reports highlighting code compliance and recommendations

Example questions:
- What are the R-value requirements for walls in climate zone 4?
- How do I calculate lighting power density for commercial buildings?
- What are the HVAC system efficiency requirements?
- What are the window U-factor limits?
- How do renewable energy requirements work in the WSEC?
""")

# Download button for analysis history
if st.session_state.messages:
    analysis_history = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.messages])
    st.download_button(
        "Download WSEC Analysis Report",
        analysis_history,
        file_name="wsec_compliance_report.txt",
        mime="text/plain"
    )