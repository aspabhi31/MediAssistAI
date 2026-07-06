import streamlit as st
from rag import process_urls, generate_answer

st.title("MediAssist AI")

url1 = st.sidebar.text_input("Search 1")
url2 = st.sidebar.text_input("Search 2")
url3 = st.sidebar.text_input("Search 3")

placeholder = st.empty()

process_url_button = st.sidebar.button("Process Searches")
if process_url_button:
    urls = [url for url in (url1, url2, url3) if url!='']
    if len(urls) == 0:
        placeholder.text("You must provide at least one valid search")
    else:
        for status in process_urls(urls):
            placeholder.text(status)

query = placeholder.text_input("Question")
if query:
    try:
        answer, sources = generate_answer(query)
        st.header("Answer:")
        st.write(answer)

        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
    except RuntimeError as e:
        placeholder.text("You must process urls first")
