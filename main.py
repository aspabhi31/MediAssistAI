import streamlit as st
from rag import process_urls, generate_answer

st.title("MediAssist AI - PubMed Q&A")

# Single search term input
search_term = st.sidebar.text_input("Enter a medical search term", 
                                    placeholder="e.g., hypertension treatment")

max_results = st.sidebar.slider("Max articles to fetch", 5, 50, 20)

placeholder = st.empty()

process_button = st.sidebar.button("Fetch & Index Articles")
if process_button and search_term:
    # Process search term
    for status in process_urls(search_term, max_results=max_results):
        placeholder.text(status)

query = st.text_input("Ask a question about the articles")
if query:
    try:
        answer, sources = generate_answer(query)
        st.header("Answer:")
        st.write(answer)

        if sources:
            st.subheader("Sources:")
            for source in sources.split("; "):
                st.write(source)
    except RuntimeError as e:
        placeholder.text("You must fetch articles first (click the button)")
