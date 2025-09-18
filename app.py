from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from utils import get_img_base64

st.set_page_config(
    page_title="RAG Desk · AI Customer Support",   # ← 这里就是页面标题
    page_icon="img/small_logo.png",                # 可换成你的小图标；不存在就删掉这一行
    layout="wide"
)

def app():
    st.set_page_config(page_title="智能客服", page_icon="img/small_logo.png", layout="wide")

    with st.sidebar:
        st.logo(
            "img/large_logo.png",  # 大一点的彩色图标
            icon_image="img/small_logo.png",  # 小一点的头像图标
            size="large"  # 让顶部 logo 更大
        )
    pg = st.navigation({
        "对话": [
            st.Page("pages/rag_chat_page.py", title="智能客服", icon=":material/chat:")
        ],
        "设置": [
            st.Page("pages/knowledge_base_pase.py", title="行业知识库", icon=":material/library_books:")
        ]
    })
    pg.run()

if __name__ == "__main__":
    app()
