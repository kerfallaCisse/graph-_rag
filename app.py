from graphdb_retriever import create_rag_chain
import streamlit as st
import time


def launch_app():
    st.set_page_config(page_title="GraphRag", page_icon="🤖")
    st.title("GraphRAG with Ontotext GraphDB")
    st.write("""AI agent specialize for answering user questions about wine. The agent use a rdf knowledge graph,
             for representing knowledge about wine. When a user ask a question, the agent fetch relevants entites and find the
             bounded context of each entity as a subgraph and pass those rdf-like triples (facts) to the LLM as a context for generating response.
             """)

    if "agent_response" not in st.session_state:
        st.session_state.agent_response = ""
    if "prompt" not in st.session_state:
        st.session_state.prompt = ""

    st.session_state.prompt = st.chat_input("Ask a question")
    if st.session_state.prompt:
        with st.chat_message("user"):
            st.write(st.session_state.prompt)
            with st.spinner("Processing"):
                time.sleep(0.5)
                st.session_state.agent_response = create_rag_chain(
                    user_question=st.session_state.prompt)
        if st.session_state.agent_response != "":
            html_file = open("subgraph.html", "r", encoding="utf-8")
            source_code = html_file.read()
            st.components.v1.html(source_code, height=600)
            with st.chat_message("assistant"):
                response = st.session_state.agent_response
                st.write_stream(response_generator(response))


def response_generator(response: str):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


if __name__ == "__main__":

    launch_app()
