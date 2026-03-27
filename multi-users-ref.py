"""
PDF 기반 멀티유저 멀티세션 RAG 챗봇
- Supabase 멀티유저 로그인, 세션 저장/로드, 벡터 DB, 스트리밍 답변
- API 키는 사이드바 상단에서 입력 (멀티유저 환경용)
실행: streamlit run multi-users-ref.py
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Generator

import streamlit as st

# .env 로드 (로컬 개발용, Streamlit Cloud에서는 Secrets 사용)
for base in [Path(__file__).resolve().parent.parent.parent, Path(__file__).resolve().parent, Path.cwd()]:
    env_path = base / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except Exception:
            pass
        break

# 상수
LLM_OPTIONS = ["gpt-4o-mini", "claude-sonnet-4-5", "gemini-3-pro-preview"]
CHATBOT_TITLE = "PDF 기반 멀티유저 멀티세션 RAG 챗봇"
EMBED_DIM = 1536
BATCH_SIZE = 10
RAG_TOP_K = 5


def get_supabase():
    """SUPABASE_URL, SUPABASE_ANON_KEY(또는 SUPABASE_SERVICE_ROLE_KEY)는 os.getenv로 읽음 (Streamlit Cloud Secrets)"""
    from supabase import create_client
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL, SUPABASE_ANON_KEY(또는 SUPABASE_SERVICE_ROLE_KEY) 필요. 앱 설정 → Secrets에 등록하세요.")
    return create_client(url, key)


def get_embeddings(api_key: str | None):
    """OpenAI Embeddings. api_key는 사이드바에서 입력한 값 사용."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    def embed(texts: list[str]) -> list[list[float]]:
        out = client.embeddings.create(model="text-embedding-3-small", input=texts)
        return [e.embedding for e in out.data]
    return embed


def list_sessions(supabase, user_id: str) -> list[dict]:
    r = supabase.table("chat_sessions").select("id,title,created_at").eq("user_id", user_id).order("created_at", desc=True).execute()
    return r.data or []


def load_messages(supabase, session_id: str) -> list[dict]:
    r = supabase.table("chat_messages").select("role,content").eq("session_id", session_id).order("created_at").execute()
    return [{"role": x["role"], "content": x["content"]} for x in (r.data or [])]


def save_messages(supabase, session_id: str, messages: list[dict]) -> None:
    for m in messages:
        if m.get("role") not in ("user", "assistant"):
            continue
        supabase.table("chat_messages").insert({
            "session_id": session_id,
            "role": m["role"],
            "content": m.get("content") or "",
        }).execute()


def create_session_with_title(supabase, title: str, user_id: str) -> str:
    r = supabase.table("chat_sessions").insert({"title": title, "user_id": user_id}).execute()
    return r.data[0]["id"]


def delete_session(supabase, session_id: str) -> None:
    supabase.table("vector_documents").delete().eq("session_id", session_id).execute()
    supabase.table("chat_messages").delete().eq("session_id", session_id).execute()
    supabase.table("chat_sessions").delete().eq("id", session_id).execute()


def list_vector_files(supabase, session_id: str | None = None) -> list[str]:
    q = supabase.table("vector_documents").select("file_name")
    if session_id:
        q = q.eq("session_id", session_id)
    r = q.execute()
    names = list({x["file_name"] for x in (r.data or []) if x.get("file_name")})
    return sorted(names)


def generate_session_title(first_question: str, first_answer: str, model_key: str, openai_key: str | None, anthropic_key: str | None, google_key: str | None) -> str:
    """첫 질문과 답변으로 세션 제목 생성."""
    system = "당신은 대화 제목을 짓는 도우미입니다. 첫 질문과 답변 요약을 반영해 한 줄 제목(한글, 30자 이내)만 출력하세요. 따옴표나 설명 없이 제목만."
    user = f"질문: {first_question[:200]}\n\n답변 요약: {first_answer[:300]}"
    return _call_llm([{"role": "system", "content": system}, {"role": "user", "content": user}], model_key, openai_key, anthropic_key, google_key).strip() or "새 세션"


def _call_llm(messages: list[dict], model_key: str, openai_key: str | None, anthropic_key: str | None, google_key: str | None) -> str:
    if "gpt" in model_key.lower():
        from openai import OpenAI
        key = openai_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            return "OPENAI_API_KEY를 입력하세요."
        client = OpenAI(api_key=key)
        r = client.chat.completions.create(model=model_key, messages=messages, temperature=0.3)
        return (r.choices[0].message.content or "").strip()
    if "claude" in model_key.lower():
        import anthropic
        key = anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            return "ANTHROPIC_API_KEY를 입력하세요."
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        chat = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
        c = anthropic.Anthropic(api_key=key)
        r = c.messages.create(model=model_key, max_tokens=2048, system=system, messages=chat)
        return (r.content[0].text if r.content else "").strip()
    if "gemini" in model_key.lower():
        import google.generativeai as genai
        key = google_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            return "GOOGLE_API_KEY를 입력하세요."
        genai.configure(api_key=key)
        model = genai.GenerativeModel(model_key)
        parts = "\n\n".join([f"{m['role']}: {m['content']}" for m in messages])
        r = model.generate_content(parts)
        return (r.text or "").strip()
    key = openai_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return "OPENAI_API_KEY를 입력하세요."
    from openai import OpenAI
    client = OpenAI(api_key=key)
    r = client.chat.completions.create(model=model_key, messages=messages, temperature=0.3)
    return (r.choices[0].message.content or "").strip()


def _stream_llm(messages: list[dict], model_key: str, openai_key: str | None, anthropic_key: str | None, google_key: str | None) -> Generator[str, None, None]:
    if "gpt" in model_key.lower():
        from openai import OpenAI
        key = openai_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            yield "OPENAI_API_KEY를 입력하세요."
            return
        client = OpenAI(api_key=key)
        stream = client.chat.completions.create(model=model_key, messages=messages, temperature=0.3, stream=True)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
        return
    if "claude" in model_key.lower():
        import anthropic
        key = anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            yield "ANTHROPIC_API_KEY를 입력하세요."
            return
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        chat = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
        c = anthropic.Anthropic(api_key=key)
        with c.messages.stream(model=model_key, max_tokens=2048, system=system, messages=chat) as stream:
            for text in stream.text_stream:
                yield text
        return
    if "gemini" in model_key.lower():
        import google.generativeai as genai
        key = google_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            yield "GOOGLE_API_KEY를 입력하세요."
            return
        genai.configure(api_key=key)
        model = genai.GenerativeModel(model_key)
        parts = "\n\n".join([f"{m['role']}: {m['content']}" for m in messages])
        r = model.generate_content(parts, stream=True)
        for chunk in r:
            if chunk.text:
                yield chunk.text
        return
    key = openai_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        yield "OPENAI_API_KEY를 입력하세요."
        return
    from openai import OpenAI
    client = OpenAI(api_key=key)
    stream = client.chat.completions.create(model=model_key, messages=messages, temperature=0.3, stream=True)
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def retrieve_docs(supabase, embed_fn, query: str, session_id: str | None, top_k: int = RAG_TOP_K) -> list:
    from langchain_core.documents import Document
    vec = embed_fn([query])[0]
    payload = {"query_embedding": vec, "match_count": top_k, "filter_session_id": session_id}
    r = supabase.rpc("match_vector_documents", payload).execute()
    docs = []
    for row in (r.data or []):
        docs.append(Document(page_content=row.get("content") or "", metadata={"file_name": row.get("file_name"), "session_id": row.get("session_id")}))
    return docs


def ingest_file(supabase, embed_fn, file_content: bytes, file_name: str, session_id: str) -> None:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text = ""
    if file_name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        import io
        reader = PdfReader(io.BytesIO(file_content))
        for p in reader.pages:
            text += p.extract_text() or ""
    else:
        text = file_content.decode("utf-8", errors="replace")
    if not text.strip():
        return
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        embs = embed_fn(batch)
        for content, embedding in zip(batch, embs):
            supabase.table("vector_documents").insert({
                "content": content,
                "metadata": {},
                "embedding": embedding,
                "file_name": file_name,
                "session_id": session_id,
            }).execute()


def build_rag_prompt(query: str, docs: list) -> str:
    if not docs:
        return ""
    ctx = "\n\n---\n\n".join(d.page_content for d in docs)
    return f"""아래 참고 문서를 바탕으로 질문에 답변하세요. 문서에 없는 내용은 추측하지 말고 '문서에 없습니다'라고 하세요.

참고 문서:
{ctx}

질문: {query}

답변 마지막에 반드시 '추가로 물어볼 만한 질문' 제목 아래에 이 대화를 이어가기 좋은 질문 3개를 번호와 함께 제시하세요."""


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    if "working_session_id" not in st.session_state:
        st.session_state.working_session_id = str(uuid.uuid4())
    if "model_key" not in st.session_state:
        st.session_state.model_key = LLM_OPTIONS[0]
    if "openai_key" not in st.session_state:
        st.session_state.openai_key = ""
    if "anthropic_key" not in st.session_state:
        st.session_state.anthropic_key = ""
    if "google_key" not in st.session_state:
        st.session_state.google_key = ""


def render_login(supabase) -> str | None:
    """로그인/회원가입 폼 렌더링. 성공 시 user_id 반환, 실패 시 None."""
    tab_login, tab_signup = st.tabs(["로그인", "회원가입"])

    with tab_login:
        st.subheader("로그인")
        login_id = st.text_input("Login ID (이메일)", key="login_id", placeholder="user@example.com")
        password = st.text_input("비밀번호", type="password", key="login_password")
        if st.button("로그인"):
            if not login_id or not password:
                st.warning("Login ID와 비밀번호를 입력하세요.")
            else:
                try:
                    r = supabase.auth.sign_in_with_password({"email": login_id.strip(), "password": password})
                    if r.user:
                        st.session_state.user_id = str(r.user.id)
                        st.session_state.user_email = r.user.email or login_id
                        st.success("로그인되었습니다.")
                        st.rerun()
                except Exception as e:
                    st.error(f"로그인 실패: {e}")

    with tab_signup:
        st.subheader("회원가입")
        signup_email = st.text_input("이메일", key="signup_email", placeholder="user@example.com")
        signup_password = st.text_input("비밀번호", type="password", key="signup_password")
        signup_password_confirm = st.text_input("비밀번호 확인", type="password", key="signup_password_confirm")
        if st.button("가입하기"):
            if not signup_email or not signup_password:
                st.warning("이메일과 비밀번호를 입력하세요.")
            elif signup_password != signup_password_confirm:
                st.warning("비밀번호가 일치하지 않습니다.")
            elif len(signup_password) < 6:
                st.warning("비밀번호는 6자 이상이어야 합니다.")
            else:
                try:
                    r = supabase.auth.sign_up({"email": signup_email.strip(), "password": signup_password})
                    if r.user:
                        # 세션이 있으면(이메일 확인 비활성화) 바로 로그인 처리
                        if r.session:
                            st.session_state.user_id = str(r.user.id)
                            st.session_state.user_email = r.user.email or signup_email
                            st.success("회원가입되었습니다. 로그인되었습니다.")
                            st.rerun()
                        else:
                            st.success("회원가입되었습니다. 이메일 확인 링크를 확인한 뒤 로그인 탭에서 로그인하세요.")
                            st.rerun()
                    else:
                        st.success("회원가입 요청이 완료되었습니다. 이메일 확인 후 로그인하세요.")
                        st.rerun()
                except Exception as e:
                    err_msg = str(e).lower()
                    if "already registered" in err_msg or "already exists" in err_msg:
                        st.error("이미 가입된 이메일입니다. 로그인 탭에서 로그인하세요.")
                    else:
                        st.error(f"회원가입 실패: {e}")
    return None


def main():
    st.set_page_config(page_title=CHATBOT_TITLE, layout="wide")
    init_session_state()

    try:
        supabase = get_supabase()
    except Exception as e:
        st.error(f"Supabase 연결 실패: {e}. SUPABASE_URL, SUPABASE_ANON_KEY를 확인하세요.")
        st.stop()

    # ----- 로그인 확인 -----
    user_id = getattr(st.session_state, "user_id", None)
    if not user_id:
        try:
            sess = supabase.auth.get_session()
            if sess and sess.session and sess.user:
                st.session_state.user_id = str(sess.user.id)
                st.session_state.user_email = sess.user.email or ""
                user_id = st.session_state.user_id
        except Exception:
            pass

    if not user_id:
        st.title(CHATBOT_TITLE)
        render_login(supabase)
        st.caption("로그인 또는 회원가입 탭에서 이메일로 가입·로그인하세요.")
        st.stop()

    # 로그인 후: API 키는 사이드바 상단에서 입력 (아래 사이드바 블록에서 갱신됨)
    embed_key = st.session_state.get("openai_key") or os.environ.get("OPENAI_API_KEY") or ""
    try:
        embed_fn = get_embeddings(embed_key) if embed_key else None
    except Exception:
        embed_fn = None

    sessions = list_sessions(supabase, user_id)
    session_options = ["(새 대화)"] + [f"{s['title']} ({s['id'][:8]})" for s in sessions]
    session_id_by_index = [None] + [s["id"] for s in sessions]
    st.session_state["session_id_by_index"] = session_id_by_index
    try:
        default_sel = session_id_by_index.index(st.session_state.current_session_id) if st.session_state.current_session_id in session_id_by_index else 0
    except ValueError:
        default_sel = 0

    def on_session_select():
        idx = st.session_state.get("session_select", 0)
        by_index = st.session_state.get("session_id_by_index") or []
        if 0 <= idx < len(by_index) and by_index[idx] is not None:
            st.session_state.pending_load_session_id = by_index[idx]

    if st.session_state.get("pending_load_session_id"):
        sid = st.session_state.pending_load_session_id
        st.session_state.pending_load_session_id = None
        msgs = load_messages(supabase, sid)
        st.session_state.messages = msgs
        st.session_state.current_session_id = sid
        st.session_state.working_session_id = sid
        st.rerun()

    # ----- 사이드바 -----
    with st.sidebar:
        st.title("설정")
        st.caption(f"로그인: {getattr(st.session_state, 'user_email', '')}")
        if st.button("로그아웃"):
            try:
                supabase.auth.sign_out()
            except Exception:
                pass
            keys_to_clear = ["user_id", "user_email", "messages", "current_session_id", "working_session_id", "model_key", "openai_key", "anthropic_key", "google_key", "session_id_by_index", "pending_load_session_id", "show_vectordb"]
            for key in keys_to_clear:
                st.session_state.pop(key, None)
            st.rerun()

        st.divider()
        st.subheader("API 키 (필요 시 입력)")
        st.session_state.openai_key = st.text_input("OpenAI API Key", value=st.session_state.get("openai_key", ""), type="password", key="sb_openai_key")
        st.session_state.anthropic_key = st.text_input("Anthropic API Key", value=st.session_state.get("anthropic_key", ""), type="password", key="sb_anthropic_key")
        st.session_state.google_key = st.text_input("Google(Gemini) API Key", value=st.session_state.get("google_key", ""), type="password", key="sb_google_key")
        st.divider()

        model_key = st.selectbox("LLM 모델", options=LLM_OPTIONS, index=LLM_OPTIONS.index(st.session_state.model_key) if st.session_state.model_key in LLM_OPTIONS else 0)
        st.session_state.model_key = model_key

        st.subheader("세션 관리")
        sel_idx = st.selectbox(
            "세션 선택",
            range(len(session_options)),
            format_func=lambda i: session_options[i],
            index=default_sel,
            key="session_select",
            on_change=on_session_select,
        )
        selected_session_id = session_id_by_index[sel_idx]

        col1, col2 = st.columns(2)
        with col1:
            if st.button("세션로드"):
                if selected_session_id:
                    msgs = load_messages(supabase, selected_session_id)
                    st.session_state.messages = msgs
                    st.session_state.current_session_id = selected_session_id
                    st.session_state.working_session_id = selected_session_id
                    st.success("세션을 불러왔습니다.")
                    st.rerun()
                else:
                    st.warning("저장된 세션을 선택하세요.")

        with col2:
            if st.button("세션삭제"):
                sid = st.session_state.current_session_id or selected_session_id
                if sid:
                    delete_session(supabase, sid)
                    if st.session_state.current_session_id == sid:
                        st.session_state.messages = []
                        st.session_state.current_session_id = None
                        st.session_state.working_session_id = str(uuid.uuid4())
                    st.success("세션이 삭제되었습니다.")
                    st.rerun()
                else:
                    st.warning("삭제할 세션을 선택하세요.")

        if st.button("세션저장"):
            if not st.session_state.messages:
                st.warning("저장할 대화가 없습니다.")
            else:
                user_msgs = [m for m in st.session_state.messages if m.get("role") == "user"]
                asst_msgs = [m for m in st.session_state.messages if m.get("role") == "assistant"]
                first_q = user_msgs[0].get("content", "")[:200] if user_msgs else ""
                first_a = asst_msgs[0].get("content", "")[:300] if asst_msgs else ""
                title = generate_session_title(first_q, first_a, model_key, st.session_state.get("openai_key"), st.session_state.get("anthropic_key"), st.session_state.get("google_key"))
                new_id = create_session_with_title(supabase, title, user_id)
                save_messages(supabase, new_id, st.session_state.messages)
                try:
                    supabase.table("vector_documents").update({"session_id": new_id}).eq("session_id", st.session_state.working_session_id).execute()
                except Exception:
                    pass
                st.session_state.current_session_id = new_id
                st.session_state.working_session_id = new_id
                st.success(f"세션 저장: {title}")
                st.rerun()

        if st.button("화면초기화"):
            st.session_state.messages = []
            st.session_state.current_session_id = None
            st.session_state.working_session_id = str(uuid.uuid4())
            st.success("화면을 초기화했습니다.")
            st.rerun()

        st.divider()
        if st.button("vectordb"):
            st.session_state.show_vectordb = getattr(st.session_state, "show_vectordb", False) ^ True
        if getattr(st.session_state, "show_vectordb", False):
            sid = st.session_state.current_session_id or st.session_state.working_session_id
            files = list_vector_files(supabase, sid)
            if files:
                st.caption("벡터 DB 파일명:")
                for f in files:
                    st.text(f"• {f}")
            else:
                st.caption("벡터 DB에 파일이 없습니다.")

    # ----- 메인: 제목, 캡션, 채팅 -----
    st.title(CHATBOT_TITLE)
    st.caption("문서를 업로드하고 질문하세요. 세션을 저장하면 대화와 벡터가 Supabase에 유지됩니다.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg.get("content") or "")

    if not embed_fn:
        st.warning("사이드바에서 OpenAI API Key를 입력하면 문서 검색이 가능합니다.")

    up = st.file_uploader("문서 업로드 (PDF, TXT)", type=["pdf", "txt"], key="file_upload")
    if up:
        with st.spinner("문서 처리 중..."):
            try:
                if embed_fn:
                    ingest_file(supabase, embed_fn, up.getvalue(), up.name, st.session_state.working_session_id)
                    st.success(f"'{up.name}' 처리 완료.")
                else:
                    st.error("OpenAI API Key를 입력한 후 다시 시도하세요.")
            except Exception as e:
                st.error(f"처리 실패: {e}")

    prompt = st.chat_input("질문을 입력하세요")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            session_id = st.session_state.current_session_id or st.session_state.working_session_id
            docs = []
            if embed_fn:
                docs = retrieve_docs(supabase, embed_fn, prompt, session_id)
            context_prompt = build_rag_prompt(prompt, docs)
            if context_prompt:
                system = "당신은 업로드된 문서 기반 RAG 어시스턴트입니다. 참고 문서만 사용해 답변하고, 마지막에 '추가로 물어볼 만한 질문' 3개를 제시하세요."
            else:
                system = "일반 질의에 친절히 답변하세요. 가능하면 마지막에 '추가로 물어볼 만한 질문' 3개를 제시하세요."
            messages = [{"role": "system", "content": system}]
            for m in st.session_state.messages:
                if m.get("role") in ("user", "assistant"):
                    messages.append({"role": m["role"], "content": m.get("content") or ""})
            if context_prompt:
                messages = messages[:-1] + [{"role": "user", "content": context_prompt}]

            full_reply = ""
            try:
                for chunk in _stream_llm(messages, model_key, st.session_state.get("openai_key"), st.session_state.get("anthropic_key"), st.session_state.get("google_key")):
                    full_reply += chunk
                    st.markdown(full_reply + "▌")
                st.markdown(full_reply)
            except Exception as e:
                full_reply = f"오류: {e}"
                st.error(full_reply)

        st.session_state.messages.append({"role": "assistant", "content": full_reply})
        st.rerun()


if __name__ == "__main__":
    main()