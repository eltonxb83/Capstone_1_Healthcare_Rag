#---------------------------------------
# Date          : 31 Jan 26
#Author         : Elton Tay, Chatgpt as part of AI driven solution
#Dependencies   : streamlit
#Purpose        : Creating a HMI interface with point, click and type interface
#Output         : Web based application 
#Notes          : -
#----------------------------------------
import streamlit as st
from rag_pipeline import run_rag

# importing of streamlit library
# from rag_pipeline script, import code block run_rag


st.set_page_config(
    page_title = "Medical RAG Assistant",
    page_icon ="🩺",
    layout = "centered"
)

st.title("🩺 Medical RAG Assistant")
st.write("Ask a question grounded in the indexed medical guidelines.")

# Display page layout configuration

# Session State (Chat Memory)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# if "chat_history" has not being created in st.session_state, create 
# st.session_state is memory state of streamlit
# ...chat_history -> assigned named of memory

#Input
query= st.chat_input("Enter your question:")

# st.chat_input -> create field for user input

#Query handling
if query and (len(st.session_state.chat_history) ==0 or st.session_state.chat_history[-1]["content"] !=query):
    with st.spinner("Searching medical knowledge..."):
        result = run_rag(query)

        st.session_state.chat_history.append(
            {"role":"user", "content": query}
        )
        st.session_state.chat_history.append(
            {"role":"assistant", "content": result}
        )

# when user enters query-
#     .triggers st.spinner
#     .invokes run_rag code block
#     . result gets assigned "answer": answer and "chunks":retrieved dictionary list
    
#     . "query" value gets assigned to key "content", with corresponding "user" value gets assigned to "role"
# when LLM response-
    
#     . "result" value gets assigned to key "content", with corresponding "assistant" value gets assigned to "role"

# final session_state.chat_history gets two entries -
#     {"role":"user", "content": query}
#     {"role":"assistant", "content": result}

# further entries gets appended with continuation of user inputs

#Display Chat History
for idx, msg in enumerate(st.session_state.chat_history):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg['content'])

# memory state gets assigned to "idx" / "msg" and looped
# when dictionary comes with "user" value for key "role", display "content" value

    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"]["answer"])
 
# same as above but for dictionary with "assistant" value for key "role"


            #Retrieved Source Toggle
        show_chunks = st.checkbox (
            "Show retrieved sources",
            key = f"chk_{idx}"
            )

# st.checkbox generates a checkbox for user to toggle

        if show_chunks:
            for i, c in enumerate(msg["content"].get("chunks",[])):

                source = c["metadata"].get("source","unknown")
                section = c["metadata"].get("section", "")
                population =c["metadata"].get("population","")
                distance = c.get("distance", None)
                title = f"Source {i+1} | {source}"

                if section:
                    title += f" | {section}"

                with st.expander(title):
                    if distance is not None:
                        st.caption(f"Similarity distance {distance:.4f}")
                        
                    if population:
                        st.caption(f"Population: {population}")
                        st.write(c["text"])

                    if source.endswith(".pdf"):
                        st.markdown(f"Reference file : `{source}`")

# if show_chunk gets toggled-
#     . enumerate list msg's [content] -i  & [chunk] - c columns
#     . assigning metadata (generated in retrieve code block from retrievel_faiss script) and displaying them

with st.sidebar:
    st.header("Controls")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []

st.divider()
st.caption("Powered by FAISS + Local LLM + Streamlit")