import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agent import agent

st.set_page_config(page_title="AI Social Media Post Creator", layout="wide")
st.title("AI Social Media Post Creator 📱")
st.markdown(
    """
    **Description:** Generate platform-optimized social media posts (Instagram, LinkedIn, X/Twitter, TikTok), 
    including captions, hashtags, and suggested posting schedules.
    """
)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []
if "tool_calls" not in st.session_state:
    st.session_state["tool_calls"] = []

chat_col, info_col = st.columns([3, 1])

with chat_col:
    st.subheader("Chat with your AI Social Media Assistant")
    
    # Display conversation
    for m in st.session_state["conversation"]:
        if isinstance(m, HumanMessage):
            st.markdown(f"**You:** {m.content}")
        elif isinstance(m, ToolMessage):
            st.markdown(f"**🔧 Tool [{m.tool_call_id}]:** {m.content}")
        else:
            st.markdown(f"**AI:** {m.content}")

    user_input = st.text_area(
        "Describe the post you want (platform, topic, tone, etc.):",
        height=100,
        key="user_input"
    )

    if st.button("Generate Post"):
        if user_input.strip():
            st.session_state["conversation"].append(HumanMessage(content=user_input.strip()))
            try:
                result = agent.invoke({"messages": st.session_state["conversation"]})
                st.session_state["conversation"] = result["messages"]

                # Track tool calls if any
                for m in result["messages"]:
                    if isinstance(m, ToolMessage):
                        st.session_state["tool_calls"].append(
                            {"tool": m.tool_call_id, "result": m.content}
                        )
            except Exception as e:
                st.error(f"Agent error: {e}")

            st.session_state.pop("user_input", None)
            st.rerun()

with info_col:
    st.subheader("Tool Calls / Outputs")
    if not st.session_state["tool_calls"]:
        st.info("No tool calls yet.")
    else:
        for idx, rec in enumerate(reversed(st.session_state["tool_calls"])):
            st.write(f"{idx+1}. {rec['tool']} -> {rec['result']}")
