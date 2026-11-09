"""The client: a simple chat screen.

    streamlit run app.py
"""
import streamlit as st

from rag import RagBot

st.set_page_config(page_title="בוט נהלים")
st.markdown("<style>.stApp {direction: rtl; text-align: right;}</style>", unsafe_allow_html=True)
st.title("בוט נהלי הארגון")


@st.cache_resource  # load the bot once, not on every message
def load_bot():
    return RagBot()


try:
    bot = load_bot()
except Exception as error:
    st.error(f"הבוט לא נטען. האם הרצת python ingest.py? פרטים: {error}")
    st.stop()

question = st.chat_input("שאלי על הנהלים...")
if question:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("מחפש ועונה..."):
                answer, chunks, expanded = bot.ask(question)
            st.write(answer)
            if expanded:
                st.caption("השאלה קצרה, ולכן הורחבה לפני החיפוש")
            with st.expander("הקטעים שהבוט קיבל"):
                for chunk in chunks:
                    st.write(chunk.node.get_content())
                    st.divider()
        except Exception as error:
            if "429" in str(error):
                st.error("חרגנו ממכסת השכבה החינמית. המתיני דקה ונסי שוב.")
            else:
                st.error(f"שגיאה: {error}")
