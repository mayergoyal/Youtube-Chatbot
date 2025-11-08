from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import re 
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
#videoid="2caQ4j9oohE"  # Replace with your YouTube video ID
import subprocess
def clean_vtt_file(video_id):
    raw_text = Path(f"{video_id}.en.vtt").read_text(encoding="utf-8")


    cleaned = re.sub(r"WEBVTT[\s\S]*?(\d{2}:\d{2}:\d{2}\.\d{3})", r"\1", raw_text)
    cleaned=re.sub(r"\d{2}:\d{2}:\d{2}\.\d{3}.*\n", " ", cleaned)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Save to cleaned file
    Path(f"{video_id}_cleaned.txt").write_text(cleaned, encoding="utf-8")
    print(f"✅ Cleaned transcript saved as {video_id}_cleaned.txt")
    print(f"📝 Preview:\n{cleaned[:400]}...")
    return cleaned
    
    
def process_video(video_id):
    print(f"📹 Processing video ID: {video_id}")
    subprocess.run([
    "yt-dlp",
    "--skip-download",
    "--write-auto-subs",
    "--sub-lang", "en",
    "--sub-format", "vtt",
    "-o", f"{video_id}.%(ext)s",
    f"https://www.youtube.com/watch?v={video_id}"
])
    text=clean_vtt_file(video_id)
    # change to your video ID or file name
    

    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    #text = Path(f"{video_id}_cleaned.txt").read_text(encoding="utf-8")
    chunks=splitter.create_documents([text])
    print(len(chunks))
#print(chunks[2].page_content[:400])
    embeddings=HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store=FAISS.from_documents(chunks,embeddings)
    retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})

    llm=ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"))
    prompt=PromptTemplate(
    template="""you are a helpful assistant 
             answer  only from the provided transcript context , if the context is insufficient , just say no 
             {context}
             Question : {question}
            """,
            input_variables=["context","question"]
)

    def format_doc(rey):
        context=" ".join([doc.page_content for doc in rey])
        return context
    
    """here do cheeze ho rhin ... pehle toh context generate hoga fir question toh context retreiver se aayega and uska aise hi nhi bhej skte toh usske liye format wala function lagega and usko chain m add karne k liye use function banana hoga so isliye wo kiya h naki fir question 
   toh pehli chain m retriver chlega .. uska output fornAT_DOC M BHEJA JAYEGA
   AND QUESTION KO PASSTHROUGH KRNA HAI SINCE question ko input m dalkr output m bhi question chahiye toh it would be runnablepassthrough
   
                  """
    parallel_chain=RunnableParallel({
    'context':retriever | RunnableLambda(format_doc),
    'question':RunnablePassthrough()}
)
    parser=StrOutputParser()
    main_chain=parallel_chain|prompt|llm|parser
    return main_chain
#main_chain.invoke(" tell qualities of emma watson in pointers")

