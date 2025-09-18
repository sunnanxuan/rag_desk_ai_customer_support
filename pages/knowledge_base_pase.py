from dotenv import load_dotenv
load_dotenv()

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import streamlit as st
from utils import get_kb_names


if not st.session_state.get("_page_title_set"):
    st.set_page_config(
        page_title="RAG Desk · AI Customer Support",
        page_icon="img/small_logo.png",
        layout="wide"
    )
    st.session_state["_page_title_set"] = True

# ========= 固定的后端参数（不在网页显示）=========
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200

# ========= 知识库根目录，可按需修改 =========
KB_ROOT = Path("knowledge_bases")

# ========= 预设的 OpenAI Embedding 模型 =========
PRESET_OPENAI_EMBED_MODELS = [
    ("text-embedding-3-small · 1536 维（默认）", "text-embedding-3-small"),
    ("text-embedding-3-large · 3072 维（更高质量）", "text-embedding-3-large"),
    ("text-embedding-ada-002 · 1536 维（旧版兼容）", "text-embedding-ada-002"),
    ("自定义…", "__custom__"),
]


# =============== 基础工具函数 ===============
def _slugify(name: str) -> str:
    import re
    s = name.strip().lower()
    s = re.sub(r"[^\w\-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "kb"

def _ensure_dirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _save_uploaded_files(files, dest_dir: Path):
    saved_paths = []
    for f in files:
        filename = Path(f.name).name
        out = dest_dir / filename
        with open(out, "wb") as w:
            w.write(f.read())
        saved_paths.append(out)
    return saved_paths

def _load_texts(paths):
    """
    读取文件为纯文本；支持 .txt, .md, .pdf, .docx
    """
    docs, metas = [], []
    for p in paths:
        p = Path(p)
        ext = p.suffix.lower()
        text = ""

        try:
            if ext in [".txt", ".md"]:
                text = p.read_text(encoding="utf-8", errors="ignore")

            elif ext == ".pdf":
                try:
                    from pypdf import PdfReader  # pip install pypdf
                except Exception as e:
                    raise RuntimeError("缺少依赖 pypdf，请先安装：pip install pypdf") from e
                reader = PdfReader(str(p))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)

            elif ext == ".docx":
                try:
                    import docx  # pip install python-docx
                except Exception as e:
                    raise RuntimeError("缺少依赖 python-docx，请先安装：pip install python-docx") from e
                d = docx.Document(str(p))
                text = "\n".join(par.text for par in d.paragraphs)

            else:
                # 其他格式可按需扩展
                continue

        except Exception as e:
            st.warning(f"读取文件失败：{p.name}，原因：{e}")
            continue

        if text and text.strip():
            docs.append(text)
            metas.append({"source": str(p), "filename": p.name, "ext": ext})
    return docs, metas

def _split_texts(texts, metas, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception as e:
        raise RuntimeError(
            "缺少 langchain-text-splitters，请安装：pip install langchain-text-splitters"
        ) from e

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    )
    chunk_texts, chunk_metas = [], []
    for t, m in zip(texts, metas):
        chunks = splitter.split_text(t)
        chunk_texts.extend(chunks)
        chunk_metas.extend([m] * len(chunks))
    return chunk_texts, chunk_metas

def _get_embeddings_openai(model_name: str):
    try:
        from langchain_openai import OpenAIEmbeddings
    except Exception as e:
        raise RuntimeError("缺少 langchain-openai，请安装：pip install langchain-openai") from e
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("未检测到 OPENAI_API_KEY 环境变量，后续可能报 401。请在 .env 或系统环境中设置。")
    return OpenAIEmbeddings(model=model_name)

def _build_chroma(texts, metas, persist_dir: Path, embeddings):
    try:
        from langchain_community.vectorstores import Chroma
    except Exception as e:
        raise RuntimeError(
            "缺少 chroma 依赖，请安装：pip install chromadb langchain-community"
        ) from e

    _ensure_dirs(persist_dir)
    vs = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metas,
        persist_directory=str(persist_dir),
    )
    return vs

def _list_existing_kbs():
    # 优先用 utils.get_kb_names（与现有项目保持一致）
    try:
        names = get_kb_names()
        if isinstance(names, (list, tuple)) and names:
            return sorted(set(_slugify(n) for n in names))
    except Exception:
        pass
    if KB_ROOT.exists():
        return sorted([p.name for p in KB_ROOT.iterdir() if p.is_dir()])
    return []


def _load_meta(kb_dir: Path) -> dict:
    meta_path = kb_dir / "meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_meta(kb_dir: Path, meta: dict):
    meta_path = kb_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def knowledge_base_page():
    st.title("行业知识库")
    st.caption("上传文档，构建向量库，用于对话检索（RAG）")

    _ensure_dirs(KB_ROOT)
    st.subheader("① 创建新的知识库")
    with st.form("create_kb_form", clear_on_submit=False):
        kb_name_input = st.text_input("知识库名称（将转为小写 slug）", placeholder="例如：banking_faq")
        kb_desc_input = st.text_area("知识库描述（可选）", placeholder="例：银行客服常见问题与回答汇总…", height=80)
        uploaded_files = st.file_uploader(
            "上传文档（可多选）",
            type=["txt", "md", "pdf", "docx"],
            accept_multiple_files=True,
            help="支持 .txt, .md, .pdf, .docx",
        )

        label_list = [x[0] for x in PRESET_OPENAI_EMBED_MODELS]
        choice = st.selectbox("选择 OpenAI Embeddings 模型", label_list, index=0)
        mapped = dict(PRESET_OPENAI_EMBED_MODELS)[choice]
        if mapped == "__custom__":
            model_name = st.text_input(
                "自定义模型名称",
                value="text-embedding-3-small",
                help="填写任意可用的 OpenAI Embeddings 模型名",
            )
        else:
            model_name = mapped

        with st.expander("模型说明", expanded=False):
            st.markdown(
                "- **text-embedding-3-small**：1536 维，价格更低，适合大多数检索场景。\n"
                "- **text-embedding-3-large**：3072 维，更高质量，适合高精度检索/跨域语料。\n"
                "- **text-embedding-ada-002**：旧版，出于兼容保留，不建议新项目使用。\n"
                "如需其他模型，请选择“自定义…”。"
            )

        submitted = st.form_submit_button("构建知识库", use_container_width=True)

    if submitted:
        kb_name = _slugify(kb_name_input)
        if not kb_name_input.strip():
            st.error("请填写知识库名称。")
            st.stop()
        if not uploaded_files:
            st.error("请至少上传一个文档。")
            st.stop()
        if not model_name.strip():
            st.error("请选择或填写有效的 Embeddings 模型名。")
            st.stop()

        kb_dir = KB_ROOT / kb_name
        src_dir = kb_dir / "source"
        chroma_dir = kb_dir / "chroma"

        if kb_dir.exists():
            st.warning(f"知识库“{kb_name}”已存在，将覆盖其向量库（原始文件保留）。")

        _ensure_dirs(src_dir)

        with st.status("正在保存上传文件…", expanded=False) as s:
            saved_paths = _save_uploaded_files(uploaded_files, src_dir)
            s.update(label=f"已保存 {len(saved_paths)} 个文件", expanded=False)

        with st.status("正在读取与切分文档…", expanded=False) as s:
            texts, metas = _load_texts(saved_paths)
            if not texts:
                s.update(label="未能从文件中读取到文本内容。", state="error")
                st.stop()
            # 使用固定的切分参数
            chunk_texts, chunk_metas = _split_texts(
                texts, metas,
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            )
            s.update(label=f"已切分为 {len(chunk_texts)} 个片段", expanded=False)

        with st.status("正在创建向量库（Chroma）…", expanded=False) as s:
            try:
                embeddings = _get_embeddings_openai(model_name)
            except Exception as e:
                s.update(label=f"创建 Embeddings 失败：{e}", state="error")
                st.stop()
            try:
                _build_chroma(chunk_texts, chunk_metas, chroma_dir, embeddings)
            except Exception as e:
                s.update(label=f"构建 Chroma 失败：{e}", state="error")
                st.stop()
            s.update(label="向量库构建完成", expanded=False)

        meta = {
            "kb_name": kb_name,
            "description": kb_desc_input.strip(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "files": [Path(p).name for p in saved_paths],
            "num_files": len(saved_paths),
            "num_chunks": len(chunk_texts),
            # 固定的切分参数写入 meta（仅信息展示，不在 UI 暴露）
            "chunk_size": int(DEFAULT_CHUNK_SIZE),
            "chunk_overlap": int(DEFAULT_CHUNK_OVERLAP),
            "embedding_backend": "OpenAI",
            "embedding_model": model_name,
            "persist_directory": str(chroma_dir),
        }
        _save_meta(kb_dir, meta)

        st.success(f"✅ 知识库“{kb_name}”已构建完成！")
        with st.expander("构建摘要", expanded=False):
            st.json(meta, expanded=False)
        st.toast("知识库创建成功", icon="✅")

    st.divider()

    # --- ② 管理已有知识库 ---
    st.subheader("② 管理已有知识库")
    existing = _list_existing_kbs()
    if not existing:
        st.info("当前没有已存在的知识库。")
        return

    col_a, col_b = st.columns([2, 1], vertical_alignment="center")
    with col_a:
        kb_selected = st.selectbox("选择一个知识库", existing)
    with col_b:
        show_meta = st.toggle("显示元信息", value=True)

    kb_dir = KB_ROOT / kb_selected
    meta = _load_meta(kb_dir)

    # 元信息展示（包含固定的 chunk 配置，仅信息用途）
    if show_meta:
        st.caption("知识库信息")
        if meta:
            st.code(json.dumps(meta, ensure_ascii=False, indent=2), wrap_lines=True)
        else:
            st.write("未找到 meta.json。")

    # 在线编辑“描述”
    st.markdown("**编辑描述**")
    new_desc = st.text_area("知识库描述", value=meta.get("description", ""), height=80, key=f"desc_{kb_selected}")
    if st.button("保存描述", use_container_width=False):
        meta["description"] = new_desc.strip()
        meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _save_meta(kb_dir, meta)
        st.success("已保存知识库描述。")

    # 操作按钮：重建 / 删除
    c1, c2, _ = st.columns([1, 1, 6])
    with c1:
        if st.button("重建向量库", use_container_width=True):
            src_dir = kb_dir / "source"
            chroma_dir = kb_dir / "chroma"
            if not src_dir.exists() or not any(src_dir.iterdir()):
                st.error("没有找到原始文件（source/）。无法重建。")
            else:
                with st.status("正在重建向量库…", expanded=False) as s:
                    texts, metas = _load_texts([p for p in src_dir.iterdir() if p.is_file()])
                    if not texts:
                        s.update(label="原始文件中没有可读取的文本内容。", state="error")
                        st.stop()

                    # 仍然使用固定的切分参数
                    chunk_texts, chunk_metas = _split_texts(
                        texts, metas,
                        chunk_size=DEFAULT_CHUNK_SIZE,
                        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                    )

                    model_name_saved = meta.get("embedding_model", "text-embedding-3-small")

                    try:
                        embeddings = _get_embeddings_openai(model_name_saved)
                        if chroma_dir.exists():
                            shutil.rmtree(chroma_dir)
                        _build_chroma(chunk_texts, chunk_metas, chroma_dir, embeddings)
                    except Exception as e:
                        s.update(label=f"重建失败：{e}", state="error")
                        st.stop()

                    s.update(label="重建完成", expanded=False)

                meta.update({
                    "num_chunks": len(chunk_texts),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "persist_directory": str(chroma_dir),
                    # 确保 meta 中反映固定参数
                    "chunk_size": int(DEFAULT_CHUNK_SIZE),
                    "chunk_overlap": int(DEFAULT_CHUNK_OVERLAP),
                    "embedding_backend": "OpenAI",
                })
                _save_meta(kb_dir, meta)
                st.success("✅ 已重建向量库。")

    with c2:
        if st.button("删除知识库", type="secondary", use_container_width=True):
            try:
                shutil.rmtree(kb_dir)
                st.success(f"已删除知识库：{kb_selected}")
                st.toast("删除成功", icon="🗑️")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"删除失败：{e}")

# 仅当作为页面运行时才渲染；被 import（如 app.py 取 __file__）时不执行
if __name__ == "__main__":
    knowledge_base_page()
