import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # allow faiss and torch to share OpenMP on macOS
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

loader = DirectoryLoader(
    path="./papers",
    glob="**/*.pdf",
    loader_cls=UnstructuredFileLoader,   # UnstructuredFileLoader trong Langchain là class dùng để đọc data từ các loại file khác nhau: pdf, docs, excel, powerpoint, html, hình ảnh,...Sau khi đọc sẽ chuyển thành các dạng có cấu trúc hơn, để sử dụng tiếp
    show_progress=True,
    use_multithreading=True,
)

docs = loader.load()  # đọc dữ liệu vào

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",    # HEADDOWN 1-6
    "```\n",        # CODE BLOCK
    "\n\\*\\*\\*+\n", # Các đường kẻ ngang phân cách (***, ---, ___)
    "\n---+\n",
    "\n___+\n",
    "\n\n",         # Hết đoạn văn
    "\n",           # Xuống dòng
    " ",
    "",
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1200,   # chia đoạn chunk con không quá 1200 characters chứ không phải 1200 token hay từ.
    chunk_overlap=200,   # giao của các chunk, để tránh trường hợp khi cắt thì nất nghĩa
    add_start_index=True,  # lưu lại vị trí bắt đầu của mỗi chunk trong tài liệu gốc  -> để bot trả về nguồn
    strip_whitespace=True,  # loại bỏ khoảng trắng ở đầu và cuối
    separators=MARKDOWN_SEPARATORS,
    is_separator_regex=True,
)

splits = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vectorstore = FAISS.from_documents(
    documents=splits,
    embedding=embeddings,
    distance_strategy=DistanceStrategy.COSINE,
)

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k":5, "score_threshold": 0.0}
)

template = (
    "You are a strict, citation-focused assistant for a private knowledge base.\n"
    "RULES:\n"
    "1) Use ONLY the provided context to answer.\n"
    "2) If the answer is not clearly contained in the context, say: "
    "\"I don't know based on the provided documents.\"\n"
    "3) Do NOT use outside knowledge, guessing, or web information.\n"
    "4) If applicable, cite sources as (source:page) using the metadata.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(
    model = "gpt-5-mini",
    temperature=0      # mức độ ngẫu nhiên trong câu trloi của model, càng cao thì càng sáng tạo
)

rag_chain = (          # pipeline toàn bộ bước vận hành của chatbot
    {"context": retriever, "question": RunnablePassthrough()}    # nhận câu hỏi input từ người dùng, lấy ngữ cảnh liên quan từ retriever
    | prompt          # nhét toàn bộ input và context này vào trong prompt
    | llm             # gửi prompt cho llm
    | StrOutputParser()   #output
)

question = input("What is the question? ")

answer = rag_chain.invoke(question)

print(answer)