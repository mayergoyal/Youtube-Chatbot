import streamlit as st
from work import process_video

st.header("YouTube Video Chatbot")
video_link=st.text_input("enter youtube link here")
if video_link:
    video_id=video_link.split("v=")[-1]
    st.write(f"video id is {video_id}")
    st.video(video_link)
    if st.button("Now we can have a chatbot here"):
        with st.spinner("processing and fetching transcript..."):
            try:
                st.session_state["main_chain"]=process_video(video_id)
                st.success("processed successfully")
            except Exception as e:
                st.error(e)
                
if "main_chain" in st.session_state:
    userinput=st.chat_input("ask here")
    if userinput:
        st.chat_message("user").write(userinput)
        with st.spinner("thinking.."):
            ans=st.session_state["main_chain"].invoke(userinput)
        st.chat_message("assistant").write(ans)
        
        
