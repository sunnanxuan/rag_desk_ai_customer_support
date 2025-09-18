from dotenv import load_dotenv
load_dotenv()

import os,base64
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from config import *





def get_img_base64(path):
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()




def get_llm_models(platform: str) -> List[str]:
    if platform == "OpenAI":
        return [
            "gpt-4o-mini",
            "gpt-4.1-mini",
            "gpt-4o",
            "gpt-4.1",
            "gpt-3.5-turbo",
        ]
    return []


def get_chatllm(platform: str, model: str, temperature: float = 0.3):
    if platform == "OpenAI":
        return ChatOpenAI(model=model, temperature=temperature)
    raise ValueError(f"Unsupported platform: {platform}")


def get_kb_names() -> List[str]:
    root = os.path.dirname(os.path.dirname(__file__))
    kb_root = os.path.join(root, "kb")
    names: List[str] = []
    if os.path.isdir(kb_root):
        for entry in os.listdir(kb_root):
            path = os.path.join(kb_root, entry)
            if os.path.isdir(path):
                names.append(entry)
    return sorted(names)


def get_embedding_model(platform_type: str = "OpenAI"):
    if platform_type == "OpenAI":
        return OpenAIEmbeddings(model="text-embedding-3-small")
    raise ValueError(f"Unsupported platform_type: {platform_type}")
