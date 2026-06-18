""" Task 4: Creating an Interactive Chat Interface
To empower non-technical support, compliance, and product managers, you will encapsulate your 
backend RAG pipeline into an interactive web UI.
Step-by-Step Implementation Protocol:
•	UI Framework Selection: Build the application interface in `app.py` utilizing Streamlit or 
Gradio. Streamlit is highly recommended for multi-column analytic dashboard layouts,
while Gradio fits perfectly into conversational interfaces.
•	Conversational & Input Components: Provide a clean, full-width text input box or a dedicated 
chat interface (`st.chat_input` or `gr.ChatInterface`). Implement a responsive submit button and 
a separate 'Clear Conversation' option to flush system state memory.
•	Source & Evidence Display: Crucial for enterprise trust! Design an expanding or dedicated 
secondary metadata container below or beside the AI answer. Loop over the retrieved source chunks, displaying the raw text narrative snippet, the actual `complaint_id`, and its corresponding sub-issue category.
•	Response Streaming (Recommended): Configure token-by-token generation streaming using g
enerator loops. This provides visual feedback and eliminates latency frustration for business stakeholders.
 """
 
 