import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import glob

# تحميل المتغيرات البيئية
load_dotenv()

# إعداد الصفحة
st.set_page_config(
    page_title="شات بوت الذكي",
    page_icon="🤖",
    layout="centered"
)

# العنوان الرئيسي
st.title("🤖 شات بوت الذكي")
st.markdown("اسألني أي سؤال وسأجيبك بناءً على محتوى الملفات النصية!")

# إعداد API Key
def setup_api_key():
    api_key = st.sidebar.text_input("أدخل Gemini API Key:", type="password")
    if api_key:
        os.environ['GEMINI_API_KEY'] = api_key
        return api_key
    return None

# تحميل وقراءة الملفات النصية
def load_text_files():
    text_data = ""
    text_files = glob.glob("data/*.txt")
    
    if not text_files:
        st.warning("⚠️ لم يتم العثور على ملفات نصية في مجلد 'data/'")
        return ""
    
    for file_path in text_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text_data += f"\n\n--- محتوى ملف {os.path.basename(file_path)} ---\n"
                text_data += file.read()
        except Exception as e:
            st.error(f"خطأ في قراءة الملف {file_path}: {str(e)}")
    
    return text_data

# تهيئة نموذج Gemini
def initialize_model(api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        return model
    except Exception as e:
        st.error(f"خطأ في تهيئة النموذج: {str(e)}")
        return None

# إنشاء الرد بناءً على محتوى الملفات
def generate_response(model, context, question):
    prompt = f"""
    بناءً على المحتوى التالي من الملفات النصية:
    
    {context}
    
    السؤال: {question}
    
    يرجى الإجابة على السؤال بناءً فقط على المعلومات الموجودة في الملفات النصية أعلاه.
    إذا لم تكن الإجابة موجودة في النص، قل بكل أدب أنك لا تملك المعلومات الكافية.
    أجب باللغة العربية ما لم يطلب منك غير ذلك.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"حدث خطأ أثناء توليد الرد: {str(e)}"

# الواجهة الرئيسية
def main():
    # الشريط الجانبي
    st.sidebar.header("الإعدادات")
    api_key = setup_api_key()
    
    # تحميل الملفات النصية
    st.sidebar.subheader("الملفات المتاحة")
    text_files = glob.glob("data/*.txt")
    for file in text_files:
        st.sidebar.write(f"📄 {os.path.basename(file)}")
    
    if not api_key:
        st.info("👈 الرجاء إدخال Gemini API Key في الشريط الجانبي لبدء المحادثة")
        return
    
    # تحميل البيانات
    with st.spinner("جاري تحميل الملفات النصية..."):
        context_data = load_text_files()
    
    if not context_data:
        return
    
    # تهيئة النموذج
    model = initialize_model(api_key)
    if not model:
        return
    
    # عرض سجل المحادثة
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # عرض الرسائل السابقة
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # استقبال السؤال من المستخدم
    if question := st.chat_input("اكتب سؤالك هنا..."):
        # إضافة سؤال المستخدم إلى السجل
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        # توليد الرد
        with st.chat_message("assistant"):
            with st.spinner("جاري البحث في الملفات وإعداد الرد..."):
                response = generate_response(model, context_data, question)
                st.markdown(response)
        
        # إضافة رد المساعد إلى السجل
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()