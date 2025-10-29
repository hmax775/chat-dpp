import streamlit as st
import os
# تم تحديث الاستيراد ليشمل DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

# 1. إعدادات Streamlit
st.set_page_config(page_title="🤖 بوت الأسئلة المخصصة (RAG)", layout="wide")
st.title("💡 شات بوت مخصص باستخدام Streamlit و Gemini (RAG)")
st.caption("يسأل ويجيب فقط من محتوى الملفات النصية المرفوعة في مجلد 'data'.")

# التأكد من وجود مفتاح API (سيتم تحميله من Streamlit Secrets)
if "GEMINI_API_KEY" not in st.secrets:
    st.error("يرجى إضافة مفتاح Gemini API إلى Streamlit Secrets باسم 'GEMINI_API_KEY'")
else:
    # 2. تهيئة نموذج Gemini
    llm = GoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=st.secrets["GEMINI_API_KEY"],
        temperature=0.1
    )
    # نموذج التضمين لإنشاء متجهات البحث
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=st.secrets["GEMINI_API_KEY"]
    )

    # 3. معالجة الملفات النصية (RAG Pipeline)
    try:
        # **التعديل هنا:** تحميل جميع الملفات النصية (.txt) من مجلد data
        # DirectoryLoader يقوم بتحميل كل الملفات المطابقة للنمط المحدد
        loader = DirectoryLoader(
            './data',             # المسار إلى المجلد
            glob="**/*.txt",      # النمط: كل الملفات بامتداد .txt
            loader_cls=TextLoader # استخدام TextLoader لتحميل المحتوى كنص
        )
        documents = loader.load()

        # التحقق من وجود مستندات للبدء
        if not documents:
            st.warning("⚠️ مجلد 'data' فارغ. يرجى رفع ملفاتك النصية (.txt) داخله.")
        else:
            # تقسيم النص إلى أجزاء (Chunks)
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(documents)

            # إنشاء Vector Store (قاعدة بيانات المتجهات) باستخدام Chroma
            db = Chroma.from_documents(texts, embeddings)

            # 4. إعداد سلسلة الاسترجاع والإجابة (RetrievalQA Chain)
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=db.as_retriever()
            )

            # 5. منطق الشات بوت في Streamlit
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("اسألني عن محتوى ملفاتك..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # إرسال استعلام موجه للنموذج
                with st.spinner("جاري البحث في الملفات..."):
                    full_prompt = (
                        "أجب على السؤال التالي بناءً على السياق المقدم فقط. "
                        "إذا لم تجد الإجابة في السياق، أجب بـ 'المعلومة غير متوفرة في الملفات المرفوعة'."
                        f"\n\nالسؤال: {prompt}"
                    )
                    
                    response = qa.run(full_prompt)

                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)

    except Exception as e:
        st.error(f"حدث خطأ غير متوقع: {e}")
        st.caption("يرجى التحقق من وجود مجلد 'data' وملفات .txt بداخله، ومفتاح API.")
