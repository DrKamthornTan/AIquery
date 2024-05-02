import argparse
from dataclasses import dataclass
import re
from translate import Translator
import streamlit as st
import csv
import base64

from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

st.set_page_config(page_title=None, page_icon=None, layout="wide")

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    st.title("DHV Health Packages AI Query")
    st.write("แพ็คเกจเมื่อ พค. 2567")
    query_text = st.text_input("กรุณาลงข้อมูลภาษาไทยหรืออังกฤษ แล้วกด Enter")

    if query_text:
        # Google Translate
        try:
            translator = Translator(from_lang='th', to_lang='en')
            translated_text = translator.translate(query_text)
        except Exception as e:
            st.write(f"Error translating: {str(e)}")
            return

        # Rest of the code...
        st.write(f"Translated Text: {translated_text}")

        # Prepare the DB.
        openai_api_key = ""  # Replace with your actual OpenAI API key
        if not openai_api_key:
            st.write("OpenAI API key is not provided.")
            return

        embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB.
        results = db.similarity_search_with_relevance_scores(translated_text, k=3)
        if not results or (results and results[0][1] < 0.7):
            st.write("ไม่สามารถค้นหาคำตอบขณะนี้ได้ โปรดติดต่อแพทย์ของท่าน")
            return

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=translated_text)
        st.write(prompt)

        model = ChatOpenAI()
        response_text = model.predict(prompt)

        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"<span style='color:red'>{response_text}</span>\nSources: {sources}"
        st.write(formatted_response, unsafe_allow_html=True)

        import difflib
        import csv
        import matplotlib.pyplot as plt
        import os
        import IPython.display as display

        # Assuming you have a "pack.csv" file containing the data
        pack_file_path = "C:\\Users\\kamth\\QueryFull\\data\\pack.csv"
        gif_directory = "C:\\Users\\kamth\\QueryFull\\GIF"

        # Read the pack.csv file and extract the name, urls, and pix fields
        sources_dict = {}
        with open(pack_file_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                name = row["name"]
                urls = row["urls"]
                pix = row["pix"]
                sources_dict[name] = {"urls": urls, "pix": pix}

        # Assuming you want to search for similar names in the pack.csv file
        source = [doc.metadata.get("source", None) for doc, _score in results]
        matching_sources = []
        added_urls = set()  # Keep track of the added URLs
        for s in source:
            matching_names = difflib.get_close_matches(s, sources_dict.keys(), n=1, cutoff=0.8)
            if matching_names:
                matching_source = matching_names[0]
                data = sources_dict[matching_source]
                urls = data["urls"]
                pix = data["pix"]
                if urls not in added_urls:  # Check if the URL has already been added
                    matching_sources.append((matching_source, urls, pix))
                    added_urls.add(urls)

        formatted_response2 = "<span style='color:red'></span>\n"

        import streamlit.components.v1 as components     

        if matching_sources:
            formatted_response2 += "Matching packages links:\n"
            for matching_source, urls, pix in matching_sources:
                # Display the corresponding GIF image
                gif_file_name = os.path.splitext(pix)[0] + '.gif'  # Change the extension to '.gif'
                gif_file_path = os.path.join(gif_directory, gif_file_name)
                if os.path.exists(gif_file_path):
                    with open(gif_file_path, "rb") as f:
                        image_data = f.read()
                        base64_encoded = base64.b64encode(image_data).decode()
                        components.html(f'<img src="data:image/gif;base64,{base64_encoded}" alt="GIF Image" />')
                    formatted_response2 += f"<a href='{urls}'>{urls}</a>\n"
                else:
                    formatted_response2 += f"Image not found: {gif_file_path}\n"
                formatted_response2 += "\n"  # Add a new line after each URL

        else:
            formatted_response2 += "No matching sources found."

        st.write(formatted_response2, unsafe_allow_html=True)
if __name__ == "__main__":
    main()