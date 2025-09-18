from dotenv import load_dotenv
load_dotenv()


import json
import streamlit as st
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import AIMessageChunk, ToolMessage, SystemMessage
from utils import *
from tools.naive_rag_tool import get_naive_rag_tool
from prompts import *
from config import *




if not st.session_state.get("_page_title_set"):
    st.set_page_config(
        page_title="RAG Desk · AI Customer Support",
        page_icon="img/small_logo.png",
        layout="wide"
    )
    st.session_state["_page_title_set"] = True





def _list_all_kbs():
    names = set()

    # 来源 1
    try:
        for n in get_kb_names() or []:
            if n and isinstance(n, str):
                names.add(n.strip())
    except Exception:
        pass

    # 来源 2
    try:
        kb_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_bases")
        if os.path.isdir(kb_root):
            for entry in sorted(os.listdir(kb_root)):
                p = os.path.join(kb_root, entry)
                if os.path.isdir(p):
                    names.add(entry.strip())
    except Exception:
        pass

    return sorted(n for n in names if n)

# —— 顶部模型配置：默认值与动态标签 ——
def _init_cfg_defaults():
    if "cfg_platform" not in st.session_state:
        st.session_state["cfg_platform"] = PLATFORMS[0]
    models0 = get_llm_models(st.session_state["cfg_platform"])
    if "cfg_model" not in st.session_state:
        st.session_state["cfg_model"] = (models0[0] if models0 else "")
    if "cfg_temp" not in st.session_state:
        st.session_state["cfg_temp"] = 0.3
    if "cfg_hist_len" not in st.session_state:
        st.session_state["cfg_hist_len"] = 5

def _cfg_label() -> str:
    p = st.session_state.get("cfg_platform", "")
    m = st.session_state.get("cfg_model", "")
    t = st.session_state.get("cfg_temp", 0.3)
    h = st.session_state.get("cfg_hist_len", 5)
    return f"{m}"


def get_rag_graph(platform, model, temperature, selected_kbs, KBS):
    tools = [KBS[k] for k in selected_kbs] if selected_kbs else []
    tool_node = ToolNode(tools) if tools else None

    def call_model(state):
        llm = get_chatllm(platform, model, temperature=temperature)
        llm_with_tools = llm.bind_tools(tools) if tools else llm
        msgs = [SystemMessage(content=RAG_ROUTING_SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [llm_with_tools.invoke(msgs)]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    if tool_node:
        workflow.add_node("tools", tool_node)
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "agent")

    workflow.set_entry_point("agent")
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    return app

def graph_response(graph, input_messages):
    """把 LangGraph 的流式消息转成 Streamlit 可写入的生成器"""
    for event in graph.invoke(
        {"messages": input_messages},
        config={"configurable": {"thread_id": 42}},
        stream_mode="messages",
    ):
        msg = event[0]
        # Agent 文本块
        if isinstance(msg, AIMessageChunk):
            if getattr(msg, "tool_calls", None):
                st.session_state["rag_tool_calls"].append(
                    {
                        "status": "正在查询……",
                        "knowledge_base": msg.tool_calls[0]["name"].replace(
                            "_knoeledge_base_tool", ""
                        ),
                        "query": "",
                    }
                )
            yield msg.content

        # 工具返回（知识库检索结果）
        elif isinstance(msg, ToolMessage):
            status_placeholder = st.empty()
            with status_placeholder.status("正在查询……", expanded=True) as s:
                kb_name = getattr(msg, "name", "").replace(
                    "_knoeledge_base_tool", ""
                ) or "知识库"
                st.write("已调用", kb_name, "进行查询")
                content = json.loads(getattr(msg, "content", ""))
                st.write("知识库检索结果：")
                st.code(
                    content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2),
                    wrap_lines=True,
                )
                s.update(label="已完成知识库检索", expanded=False)

            # 合并到 session_state 的工具调用记录
            if len(st.session_state["rag_tool_calls"]) and "content" not in st.session_state["rag_tool_calls"][-1]:
                st.session_state["rag_tool_calls"][-1]["status"] = "已完成知识库检索！"
                st.session_state["rag_tool_calls"][-1]["content"] = content
            else:
                st.session_state["rag_tool_calls"].append(
                    {
                        "status": "已完成知识库检索！",
                        "knowledge_base": kb_name,
                        "content": content,
                    }
                )

def get_rag_chat_response(platform, model, temperature, input_messages, selected_kbs, KBS):
    app = get_rag_graph(platform, model, temperature, selected_kbs, KBS)
    return graph_response(graph=app, input_messages=input_messages)

# -----------------------------
# UI：历史记录/清空
# -----------------------------
def display_chat_history():
    for message in st.session_state["rag_chat_history_with_too_call"]:
        with st.chat_message(
            message["role"],
            avatar=get_img_base64("img/small_logo.png") if message["role"] == "assistant" else None,
        ):
            # 展示工具检索状态（如果有）
            if "tool_calls" in message:
                for tool_call in message["tool_calls"]:
                    with st.status(tool_call.get("status", "查询中"), expanded=False):
                        st.write("已调用", tool_call.get("knowledge_base", "知识库"), "进行查询")
                        if "content" in tool_call:
                            st.write("知识库查询结果：")
                            st.code(
                                tool_call["content"]
                                if isinstance(tool_call["content"], str)
                                else json.dumps(tool_call["content"], ensure_ascii=False, indent=2),
                                wrap_lines=True,
                            )
            st.write(message.get("content", ""))

def clear_chat_history():
    st.session_state["rag_chat_history"] = [
        {"role": "assistant", "content": RAG_PAGE_INTRODCCTION}
    ]
    st.session_state["rag_chat_history_with_too_call"] = [
        {"role": "assistant", "content": RAG_PAGE_INTRODCCTION}
    ]
    st.session_state["rag_tool_calls"] = []

# -----------------------------
# 页面主体
# -----------------------------
def rag_chat_page():
    # --- 构建可用的知识库工具 ---
    kbs = _list_all_kbs()
    KBS = {k: get_naive_rag_tool(k) for k in kbs}

    # --- 初始化会话状态 ---
    if "rag_chat_history" not in st.session_state:
        st.session_state["rag_chat_history"] = [
            {"role": "assistant", "content": RAG_PAGE_INTRODCCTION}
        ]
    if "rag_chat_history_with_too_call" not in st.session_state:
        st.session_state["rag_chat_history_with_too_call"] = [
            {"role": "assistant", "content": RAG_PAGE_INTRODCCTION}
        ]
    if "rag_tool_calls" not in st.session_state:
        st.session_state["rag_tool_calls"] = []

    # ================= 顶部工具条（像 ChatGPT 顶部一样） =================
    _init_cfg_defaults()  # 确保有默认值
    top_bar = st.container()
    with top_bar:
        left, mid, right = st.columns([6, 4, 2], vertical_alignment="center")
        with left:
            st.markdown("### RAG Desk · AI Customer Support")
        with mid:
            # 动态显示当前配置的弹出按钮
            with st.popover(_cfg_label(), use_container_width=True, help="配置当前对话使用的模型"):
                # 平台选择
                st.selectbox("加载方式（平台）", PLATFORMS, key="cfg_platform")

                # 根据平台刷新模型清单
                _models = get_llm_models(st.session_state["cfg_platform"])
                if _models and st.session_state.get("cfg_model") not in _models:
                    st.session_state["cfg_model"] = _models[0]
                st.selectbox("选择模型", _models, key="cfg_model")

                # 其它参数
                st.slider("Temperature", 0.1, 1.0, key="cfg_temp")
                st.slider("历史消息长度", 1, 10, key="cfg_hist_len")

        with right:
            st.button("🗑️ 清空对话", use_container_width=True, on_click=clear_chat_history)

    # 读取当前配置
    platform = st.session_state["cfg_platform"]
    model = st.session_state["cfg_model"]
    temperature = float(st.session_state["cfg_temp"])
    history_len = int(st.session_state["cfg_hist_len"])

    # --- 侧边栏：知识库选择 ---
    with st.sidebar:
        if kbs:
            selected_kbs = st.multiselect("请选择对话中可使用的知识库", kbs, default=kbs)
        else:
            st.info("未发现可用知识库。请先到「设置 → 行业知识库」上传并构建。")
            selected_kbs = []

        # 可选：展示最近一次工具调用状态
        if st.session_state["rag_tool_calls"]:
            st.divider()
            st.caption("最近一次知识库调用")
            last = st.session_state["rag_tool_calls"][-1]
            st.write("状态：", last.get("status", ""))
            st.write("知识库：", last.get("knowledge_base", ""))
            if "content" in last:
                with st.expander("查看返回内容"):
                    st.code(
                        last["content"]
                        if isinstance(last["content"], str)
                        else json.dumps(last["content"], ensure_ascii=False, indent=2),
                        wrap_lines=True,
                    )

    # --- 历史区 ---
    display_chat_history()

    # --- 输入区（页面底部的原生 chat_input） ---
    user_text = st.chat_input("请输入您的问题…")

    # --- 处理输入 ---
    if user_text:
        with st.chat_message("user"):
            st.write(user_text)

        st.session_state["rag_chat_history"].append({"role": "user", "content": user_text})
        st.session_state["rag_chat_history_with_too_call"].append({"role": "user", "content": user_text})

        # 截取历史长度
        hist = st.session_state["rag_chat_history"][-history_len:]

        stream_response = get_rag_chat_response(
            platform=platform,
            model=model,
            temperature=temperature,
            input_messages=hist,
            selected_kbs=selected_kbs,
            KBS=KBS,
        )

        with st.chat_message("assistant", avatar=get_img_base64("img/small_logo.png")):
            assistant_text = st.write_stream(stream_response)

        # 把最终回答写回会话（用于刷新回显）
        st.session_state["rag_chat_history"].append({"role": "assistant", "content": assistant_text})
        st.session_state["rag_chat_history_with_too_call"].append({"role": "assistant", "content": assistant_text})

# 仅当作为“页面脚本”运行时才渲染
if __name__ == "__main__":
    rag_chat_page()
