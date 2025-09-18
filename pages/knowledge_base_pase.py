from dotenv import load_dotenv
load_dotenv()

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
import streamlit as st

# ==== 你自己的工具（需在 utils.py 中提供）====
# 需要包含：get_kb_names, load_texts, split_texts, list_all_kbs,
#          save_uploaded_files, get_embedding_model, _slugify
from utils import *

from langchain_chroma import Chroma

# ------------------ 基础常量 ------------------
KB_ROOT = Path("kb")

# ------------------ 基础工具函数 ------------------
def _ensure_dirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _build_chroma(texts, metas, persist_dir: Path, embeddings):
    _ensure_dirs(persist_dir)
    vs = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metas,
        persist_directory=str(persist_dir),
    )
    return vs

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

# ------------------ 页面设置 ------------------
if not st.session_state.get("_page_title_set"):
    st.set_page_config(
        page_title="RAG Desk · AI Customer Support",
        page_icon="img/small_logo.png",
        layout="wide"
    )
    st.session_state["_page_title_set"] = True

# ================== 主页面 ==================
def knowledge_base_page():
    st.title("行业知识库")
    st.caption("上传文档，构建向量库，用于对话检索（RAG）")

    # ✅ 删除成功一次性提示（flash）
    if st.session_state.get("__kb_deleted_msg"):
        st.success(st.session_state["__kb_deleted_msg"])
        st.toast(st.session_state["__kb_deleted_msg"], icon="🗑️")
        del st.session_state["__kb_deleted_msg"]

    _ensure_dirs(KB_ROOT)

    # ---------------- ① 创建新的知识库 ----------------
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

        # ❌ 已移除 Embeddings 模型选择器

        submitted = st.form_submit_button("构建知识库", use_container_width=True)

    if submitted:
        kb_name = slugify(kb_name_input)
        if not kb_name_input.strip():
            st.error("请填写知识库名称。")
            st.stop()
        if not uploaded_files:
            st.error("请至少上传一个文档。")
            st.stop()

        kb_dir = KB_ROOT / kb_name
        src_dir = kb_dir / "source"
        vectorstore_dir = kb_dir / "vectorstore"

        if kb_dir.exists():
            st.warning(f"知识库“{kb_name}”已存在，将覆盖其向量库（原始文件保留）。")

        _ensure_dirs(src_dir)

        with st.status("正在保存上传文件…", expanded=False) as s:
            saved_paths = save_uploaded_files(uploaded_files, src_dir)
            s.update(label=f"已保存 {len(saved_paths)} 个文件", expanded=False)

        with st.status("正在读取与切分文档…", expanded=False) as s:
            texts, metas = load_texts(saved_paths)
            if not texts:
                s.update(label="未能从文件中读取到文本内容。", state="error")
                st.stop()

            # 使用 utils.split_texts（可由 .env 控制 DEFAULT_CHUNK_SIZE/OVERLAP）
            chunk_texts, chunk_metas = split_texts(texts, metas)
            s.update(label=f"已切分为 {len(chunk_texts)} 个片段", expanded=False)

        with st.status("正在创建向量库（Chroma）…", expanded=False) as s:
            try:
                embeddings = get_embedding_model("OpenAI")  # ✅ 统一使用 utils
                model_name_used = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            except Exception as e:
                s.update(label=f"创建 Embeddings 失败：{e}", state="error")
                st.stop()
            try:
                _build_chroma(chunk_texts, chunk_metas, vectorstore_dir, embeddings)
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
            # 信息展示：从 .env 读取（字符串即可）
            "chunk_size": os.getenv("DEFAULT_CHUNK_SIZE"),
            "chunk_overlap": os.getenv("DEFAULT_CHUNK_OVERLAP"),
            "embedding_backend": "OpenAI",
            "embedding_model": model_name_used,   # ✅ 记录实际使用的模型名
            "persist_directory": str(vectorstore_dir),
        }
        _save_meta(kb_dir, meta)

        st.success(f"✅ 知识库“{kb_name}”已构建完成！")
        with st.expander("构建摘要", expanded=False):
            st.json(meta, expanded=False)
        st.toast("知识库创建成功", icon="✅")

    st.divider()

    # ---------------- ② 管理已有知识库 ----------------
    st.subheader("② 管理已有知识库")

    existing_labels = list_all_kbs()  # list[str]（可包含中文）
    label_to_slug = {lbl: slugify(lbl) for lbl in existing_labels}
    sorted_labels = sorted(label_to_slug.keys())

    if not sorted_labels:
        st.info("当前没有已存在的知识库。")
        return

    col_a, col_b = st.columns([2, 1], vertical_alignment="center")
    with col_a:
        kb_label_selected = st.selectbox("选择一个知识库", sorted_labels)
    with col_b:
        show_meta = st.toggle("显示元信息", value=True)

    kb_slug = label_to_slug[kb_label_selected]
    kb_dir = KB_ROOT / kb_slug
    meta = _load_meta(kb_dir)

    # 元信息展示
    if show_meta:
        st.caption("知识库信息")
        if meta:
            st.code(json.dumps(meta, ensure_ascii=False, indent=2), wrap_lines=True)
        else:
            st.write("未找到 meta.json。")

    # 在线编辑“描述”
    st.markdown("**编辑描述**")
    new_desc = st.text_area("知识库描述", value=meta.get("description", ""), height=80, key=f"desc_{kb_slug}")
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
            vectorstore_dir = kb_dir / "vectorstore"
            if not src_dir.exists() or not any(src_dir.iterdir()):
                st.error("没有找到原始文件（source/）。无法重建。")
            else:
                with st.status("正在重建向量库…", expanded=False) as s:
                    texts, metas = load_texts([p for p in src_dir.iterdir() if p.is_file()])
                    if not texts:
                        s.update(label="原始文件中没有可读取的文本内容。", state="error")
                        st.stop()

                    chunk_texts, chunk_metas = split_texts(texts, metas)

                    try:
                        embeddings = get_embedding_model("OpenAI")
                        model_name_used = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
                        if vectorstore_dir.exists():
                            shutil.rmtree(vectorstore_dir)
                        _build_chroma(chunk_texts, chunk_metas, vectorstore_dir, embeddings)
                    except Exception as e:
                        s.update(label=f"重建失败：{e}", state="error")
                        st.stop()

                    s.update(label="重建完成", expanded=False)

                meta.update({
                    "num_chunks": len(chunk_texts),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "persist_directory": str(vectorstore_dir),
                    "chunk_size": os.getenv("DEFAULT_CHUNK_SIZE"),
                    "chunk_overlap": os.getenv("DEFAULT_CHUNK_OVERLAP"),
                    "embedding_backend": "OpenAI",
                    "embedding_model": model_name_used,  # ✅ 与实际一致
                })
                _save_meta(kb_dir, meta)
                st.success("✅ 已重建向量库。")

    with c2:
        if st.button("删除知识库", type="secondary", use_container_width=True):
            try:
                shutil.rmtree(kb_dir)
                # 写入一次性闪存提示并刷新
                st.session_state["__kb_deleted_msg"] = f"已删除知识库：{kb_label_selected}"
                st.rerun()
            except Exception as e:
                st.error(f"删除失败：{e}")

# 仅当作为页面运行时才渲染；被 import 时不执行
if __name__ == "__main__":
    knowledge_base_page()
