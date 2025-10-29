# -*- coding: utf-8 -*-
# **Knowledge Base Web Chatbot (Streamlit + Gemini API)**
#
# هذا التطبيق مصمم للعمل على منصات النشر السحابي المجانية مثل Streamlit Cloud.
# يتم قراءة الملفات من مجلد 'data' ويتم بناء الفهرس في الذاكرة عند بدء التشغيل.
#
# ملاحظة: يتم استخدام FAISS كفهرس في الذاكرة لتبسيط النشر السحابي المجاني.
#
# متطلبات التشغيل:
# 1. تثبيت المكتبات في requirements.txt
# 2. إنشاء مجلد 'data' ووضع ملفاتك الـ 400 (بصيغة .txt) بداخله.
# 3. توفير مفتاح Gemini API كـ "متغير بيئة" (Environment Variable) باسم GEMINI_API_KEY
#    على منصة النشر السحابي (مثل Streamlit Cloud).

import streamlit as st
import os
import faiss
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

# ============================ 1. الإعدادات والثوابت =============================
DATA_DIR = "./data"

# ============================ 2. تهيئة النماذج وبناء الفهرس ==========================

@st.cache_resource(show_spinner="⏳ جاري قراءة الملفات وبناء قاعدة المعرفة (قد يستغرق هذا وقتاً طويلاً لحجم 400 ملف)...")
def setup_knowledge_base():
    """تهيئة النماذج وبناء الفهرس. يتم تخزين الفهرس مؤقتاً لتبسيط النشر السحابي."""
    
    # التحقق من مفتاح API
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("❌ لم يتم العثور على مفتاح Gemini API. يرجى إضافته كمتغير بيئة.")
        return None

    # 1. إعداد LLM و Embedding Model
    Settings.llm = Gemini(model="gemini-2.5-flash", api_key=gemini_api_key, request_timeout=360.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    st.info("✅ تم تحميل النماذج. جاري قراءة المستندات.")
    
    # 2. قراءة المستندات
    try:
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        st.info(f"📄 تم العثور على {len(documents)} مستند (جزء).")
    except Exception as e:
        st.error(f"❌ خطأ في قراءة الملفات من مجلد 'data': {e}")
        st.warning("💡 تأكد من وجود مجلد 'data' وملفاتك النصية (.txt) بداخله.")
        return None

    if not documents:
        st.error("❌ لم يتم العثور على مستندات. لا يمكن بناء الشات بوت.")
        return None

    # 3. بناء الفهرس (باستخدام FAISS في الذاكرة)
    # ملاحظة: سنستخدم FAISS كحل مؤقت ومجاني للنشر السحابي،
    # حيث يتم تخزين الفهرس في ذاكرة التطبيق (Memory).
    try:
        # إنشاء فهرس FAISS
        dimension = 384  # هذا هو حجم المتجهات لنموذج all-MiniLM-L6-v2
        faiss_index = faiss.IndexFlatL2(dimension)
        vector_store = FaissVectorStore(faiss_index=faiss_index)

        # بناء الفهرس
        index = VectorStoreIndex.from_documents(
            documents,
            vector_store=vector_store,
            embed_model=Settings.embed_model,
        )
        st.success("✅ اكتمل بناء قاعدة المعرفة بنجاح.")
        return index
    except Exception as e:
        st.error(f"❌ فشل بناء الفهرس: {e}")
        return None

# ============================ 3. واجهة Streamlit ==========================

def main_app():
    st.set_page_config(page_title="شات بوت المعرفة المؤسسية", layout="wide", initial_sidebar_state="collapsed")
    
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
            html, body, [class*="st-"] { font-family: 'Cairo', sans-serif; text-align: right; direction: rtl; }
            .st-emotion-cache-18ni7ap { padding-top: 1rem; }
        </style>
        """, unsafe_allow_html=True)
        
    st.title("🤖 شات بوت المعرفة الخاص بالمؤسسة")
    st.markdown("هذا الشات بوت يجيب على أسئلتك **حصراً** بناءً على الـ 400 ملف الداخلي.")

    # تحميل الفهرس
    index = setup_knowledge_base()

    if index is None:
        st.stop()

    # تهيئة سجل المحادثات
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "أهلاً بك! أنا الشات بوت المؤسسي. كيف يمكنني مساعدتك في بياناتك الداخلية اليوم؟"})

    # عرض سجل المحادثات
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # معالجة استعلام المستخدم
    if prompt := st.chat_input("اطرح سؤالك هنا..."):
        # 1. إضافة رسالة المستخدم
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. إنشاء محرك الاستعلام وتنفيذ RAG
        with st.chat_message("assistant"):
            with st.spinner("جاري البحث في قاعدة المعرفة الخاصة (RAG)..."):
                try:
                    query_engine = index.as_query_engine(
                        similarity_top_k=5,
                        response_mode="compact"
                    )
                    
                    response = query_engine.query(prompt)

                    # صياغة الإجابة وإضافة المصادر
                    source_names = [Path(node.metadata.get('file_path', 'N/A')).name for node in response.source_nodes if node.metadata.get('file_path')]
                    
                    # عرض الإجابة
                    st.markdown(response.response)

                    # عرض المصادر
                    if source_names:
                        sources_markdown = "---  \n **📚 المصادر المرجعية المستخدمة:** \n"
                        sources_markdown += "\n".join([f"- `{name}`" for name in set(source_names)])
                        st.markdown(sources_markdown)
                        
                    # إضافة الرسالة الكاملة إلى سجل المحادثات
                    st.session_state.messages.append({"role": "assistant", "content": response.response + "\n\n" + sources_markdown})

                except Exception as e:
                    error_message = f"❌ حدث خطأ داخلي أثناء معالجة طلبك: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main_app()
