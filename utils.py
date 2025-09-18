from dotenv import load_dotenv
load_dotenv()

import os,base64
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from config import *
from pypdf import PdfReader
import docx
from pathlib import Path
from typing import Iterable, Callable, Tuple, List, Dict, Optional
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter




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
    root = os.path.dirname(__file__)
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



from functools import lru_cache

def list_all_kbs() -> list[str]:
    try:
        return sorted({
            n.strip()
            for n in (get_kb_names() or [])
            if isinstance(n, str) and n.strip()
        })
    except Exception:
        return []



def load_texts(paths: Iterable[str | Path],warn: Optional[Callable[[str], None]] = None,) -> Tuple[List[str], List[Dict]]:
    def _warn(msg: str):
        if warn:
            warn(msg)

    docs: List[str] = []
    metas: List[Dict] = []

    for p in paths:
        p = Path(p)
        ext = p.suffix.lower()
        text = ""

        try:
            if ext in (".txt", ".md"):
                text = p.read_text(encoding="utf-8", errors="ignore")

            elif ext == ".pdf":
                if PdfReader is None:
                    raise RuntimeError("缺少依赖 pypdf，请先安装：pip install pypdf")
                reader = PdfReader(str(p))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)

            elif ext == ".docx":
                if docx is None:
                    raise RuntimeError("缺少依赖 python-docx，请先安装：pip install python-docx")
                d = docx.Document(str(p))
                text = "\n".join(par.text for par in d.paragraphs)

            else:
                # 非支持类型：直接跳过
                continue

        except Exception as e:
            _warn(f"读取文件失败：{p.name}，原因：{e}")
            continue

        if text and text.strip():
            docs.append(text)
            metas.append({"source": str(p), "filename": p.name, "ext": ext})

    return docs, metas







def split_texts(texts, metas):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("DEFAULT_CHUNK_SIZE")),
        chunk_overlap=int(os.getenv("DEFAULT_CHUNK_OVERLAP")),
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    )
    chunk_texts, chunk_metas = [], []
    for t, m in zip(texts, metas):
        chunks = splitter.split_text(t)
        chunk_texts.extend(chunks)
        chunk_metas.extend([m] * len(chunks))
    return chunk_texts, chunk_metas
