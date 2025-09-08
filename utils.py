from dotenv import load_dotenv
load_dotenv()

import os
from typing import List, Optional

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# 可选：头像图片支持
try:
    from PIL import Image
except Exception:
    Image = None


# 供 UI 选择的平台列表（你的页面是当作变量用的）
PLATFORMS: List[str] = ["OpenAI"]


def get_img_base64(name: Optional[str] = None):
    """
    返回可用于 st.chat_message 的头像对象：
    - 若传入文件名且能找到 → 返回 PIL.Image
    - 否则 → 返回一个 emoji 字符串作为占位
    """
    if not name:
        return "🤖"


    here = os.path.dirname(__file__)
    candidates = [
        os.path.join(here, "img", name),
        os.path.join(here, name),
        os.path.abspath(name),
    ]
    for p in candidates:
        if os.path.isfile(p):
            if Image is not None:
                try:
                    return Image.open(p)
                except Exception:
                    break
            else:
                # 没有 PIL 时，用占位符
                return "🤖"
    return "🤖"


def get_llm_models(platform: str) -> List[str]:
    """
    根据平台给出可选模型列表（最小可用集，可按需增减）
    """
    if platform == "OpenAI":
        return [
            "gpt-4o-mini",
            "gpt-4.1-mini",
            "gpt-4o",
            "gpt-4.1",
            "gpt-3.5-turbo",  # 兜底旧型号，避免环境较旧时报错
        ]
    return []


def get_chatllm(platform: str, model: str, temperature: float = 0.3):
    """
    返回一个可被 langgraph 代理使用的 Chat LLM 对象
    """
    if platform == "OpenAI":
        # 需要 OPENAI_API_KEY（已在 load_dotenv() 读取）
        return ChatOpenAI(model=model, temperature=temperature)
    raise ValueError(f"Unsupported platform: {platform}")


def get_kb_names() -> List[str]:
    """
    扫描项目根目录下的 kb/ 子目录作为知识库名。
    若某个知识库目录下存在 vectorstore/，更佳；否则也允许显示该目录名。
    """
    root = os.path.dirname(os.path.dirname(__file__))  # 与你上面构建 persist_directory 的方式一致
    kb_root = os.path.join(root, "kb")
    names: List[str] = []
    if os.path.isdir(kb_root):
        for entry in os.listdir(kb_root):
            path = os.path.join(kb_root, entry)
            if os.path.isdir(path):
                # 如果存在 vectorstore/ 你就已经建库过；如果没有，也允许显示出来
                names.append(entry)
    return sorted(names)


def get_embedding_model(platform_type: str = "OpenAI"):
    if platform_type == "OpenAI":
        return OpenAIEmbeddings(model="text-embedding-3-small")
    raise ValueError(f"Unsupported platform_type: {platform_type}")
