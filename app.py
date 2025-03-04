import streamlit as st
from openai import OpenAI
from streamlit_js_eval import streamlit_js_eval
from langchain.memory import ConversationSummaryMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter
from IPython.display import Audio
from gtts import gTTS
import os
import subprocess
from pydub import AudioSegment
from pydub.playback import play
import pinecone
from pinecone import Pinecone, ServerlessSpec
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import pandas as pd



chat = ChatOpenAI(model_name = 'gpt-4o-mini', 
                seed=365,
                api_key=st.secrets["OPENAI_API_KEY"],
                temperature = 0,
                max_tokens = 200)


feedback_chat = ChatOpenAI(model_name = 'gpt-4o-mini', 
                    seed=365,
                    api_key=st.secrets["OPENAI_API_KEY"],
                    temperature = 0,
                    max_tokens = 400)

@chain
def memory_chain(query):

    main_chain = (
        RunnablePassthrough.assign(message_log = RunnableLambda(st.session_state.chat_memory.load_memory_variables) | itemgetter('message_log')) |
         prompt_template_officer |
         st.session_state.chat_instance |
         StrOutputParser()
    )
            
    response = main_chain.invoke({'query':query, 'qa_description':st.session_state.current_qa})
            
    st.session_state.chat_memory.save_context(inputs={'input':query}, outputs={'output':response})

    return response

def vector_db_fetch(query):
    query_embedding = st.session_state.bert_model.encode(query, show_progress_bar=False).tolist()
    query_results = index.query(
        vector=[query_embedding],
        top_k=12,
        include_metadata=True
    )

    for match in query_results['matches']:
        qa_details = match.get('metadata', {})
        qa_category = qa_details.get('category', 'N/A')
        qa_desc = qa_details.get('qa_descriptions', 'N/A')

        print(qa_category)

        if not st.session_state.asked_categories.get(qa_category):
            st.session_state.asked_categories[qa_category] = True
            return qa_desc
    
    st.warning("All possible questions from each category have already been asked.")
    return ""

st.set_page_config(page_title="VISA Interview Simulator", page_icon="👮")
st.title("US F1 VISA Interview Simulator")

if "setup_complete" not in st.session_state:
    st.session_state.setup_complete = False
if "db_check" not in st.session_state:
    st.session_state.db_check = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_message_count" not in st.session_state:
    st.session_state.user_message_count = 0
if "user_message_limit" not in st.session_state:
    st.session_state.user_message_limit = 4
if "feedback_shown" not in st.session_state:
    st.session_state.feedback_shown = False
if "feedback_generated" not in st.session_state:
    st.session_state.feedback_generated = False
if "chat_complete" not in st.session_state:
    st.session_state.chat_complete = False
if "pinecone" not in st.session_state:
    st.session_state.pinecone = Pinecone(api_key=st.secrets["PINECONE_API_KEY"], environment=st.secrets["PINECONE_ENV"])
if "bert_model" not in st.session_state:
    st.session_state.bert_model = SentenceTransformer("multi-qa-distilbert-cos-v1")
if "Index_name" not in st.session_state:
    st.session_state.Index_name = "visa-data"
if "asked_questions" not in st.session_state:
    st.session_state.asked_questions = []
if "asked_categories" not in st.session_state:
    st.session_state.asked_categories = {
        'Purpose of Study': False, 'University Choice': False, 'Academic Performance': False,
        'Financial Information': False, 'Personal Information': False, 'Academic Background': False,
        'Duration of Stay': False, 'Future Plans': False, 'Test Scores': False, 'Language Proficiency': False,
        'Documents': False, 'Employment Plans': False, 'Post-Graduation Plans': False, 'Study Motivation': False,
        'Intent to Return': False, 'Behavioral': False, 'Preparation': False, 'Family Ties': False,
        'Family Location': False, 'Family and School': False, 'Personal Life': False, 'Travel Plans': False,
        'Education System Knowledge': False, 'Career Prospects': False, 'Student Visa Justification': False,
        'Visa Denial Response': False
    }
if "current_qa" not in st.session_state:
    st.session_state.current_qa = ""
if "chat_instance" not in st.session_state:
    st.session_state.chat_instance = ChatOpenAI(model_name = 'gpt-4o-mini', 
                seed=365,
                api_key=st.secrets["OPENAI_API_KEY"],
                temperature = 0,
                max_tokens = 200)
if "feedback_chat_instance" not in st.session_state:
    st.session_state.feedback_chat_instance = ChatOpenAI(model_name = 'gpt-4o-mini', 
                    seed=365,
                    api_key=st.secrets["OPENAI_API_KEY"],
                    temperature = 0.3,
                    max_tokens = 100)
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationSummaryMemory(llm = ChatOpenAI(model_name='gpt-4o-mini', api_key=st.secrets["OPENAI_API_KEY"]), memory_key='message_log')

def complete_setup():
    st.session_state.setup_complete = True

def show_feedback():
    st.session_state.feedback_shown = True

if not st.session_state.setup_complete:
    st.subheader('Enter Personal Information', divider='rainbow')

    if "fullname" not in st.session_state:
        st.session_state["fullname"] = ""
    if "age" not in st.session_state:
        st.session_state["age"] = ""
    if "nationality" not in st.session_state:
        st.session_state["nationality"] = "Iranian"
    if "funding_method" not in st.session_state:
        st.session_state["funding_method"] = "Self-Fund"

    fullname = st.text_input(label="Full name", max_chars=40, value=st.session_state["fullname"], placeholder="(e.g. Elon Musk)" )
    
    age = st.text_input(label="Age", max_chars=2, value=st.session_state["age"], placeholder="(e.g. 53)" )

    nationality = st.text_input(label="Nationality", max_chars=30, value=st.session_state["nationality"], placeholder="(e.g. Iranian)" )

    st.session_state["funding_method"] = st.selectbox(
        "Funding Method",
        ("Self-Fund", "Student Loan", "Partial-Fund", "Full-Fund")
    )

    st.subheader('University and Major Information', divider='rainbow')

    if "university" not in st.session_state:
        st.session_state["university"] = ""
    if "major" not in st.session_state:
        st.session_state["major"] = "Computer Science"
    if "study_level" not in st.session_state:
        st.session_state["study_level"] = "Associate"


    university = st.text_input(label="University", max_chars=100, value=st.session_state["university"], placeholder="(e.g. University of Pennsylvania)")

    major = st.text_input(label="Major", max_chars=50, value=st.session_state["major"], placeholder="(e.g. Economics)" )

    st.session_state["study_level"] = st.selectbox(
        "Level of Study",
        ("Associate", "Bachelor's", "Master's", "Doctoral", "Postdoctoral")
    )

    st.subheader('Chat Settings', divider='rainbow')

    if "interview_type" not in st.session_state:
        st.session_state["interview_type"] = "Beginner"

    st.session_state["interview_type"] = st.selectbox(
        "Duration of Interview",
        ("Beginner", "Intermediate", "Advanced" )
    )

    if "tips_shown" not in st.session_state:
        st.session_state.tips_shown = False

    if "share_data" not in st.session_state:
        st.session_state.share_data = False

    col1, col2 = st.columns(2)
    with col1:
        st.session_state["tips_shown"] = st.checkbox(
            "Activate Tips (Highly Recommended for Starters)",
            key='visibility',
        )
    with col2:
        st.session_state["share_data"] = st.checkbox(
            "Check this if you agree to share your chat data to improve the AI, Thank you!",
            key='visibility2',
        )

    should_disable_button = (nationality == "" or 
                             fullname == "" or 
                             age == "" or 
                             university == "" or 
                             major == "" or
                             st.session_state["funding_method"] == "" or
                             st.session_state["study_level"] == "" or
                             st.session_state["interview_type"] == "")

    if st.button("Start Interview", on_click=complete_setup, disabled=should_disable_button):
        st.write("Setup complete. Starting Interview...")

if (st.session_state.setup_complete 
    and not st.session_state.feedback_shown 
    and not st.session_state.chat_complete):

    user_details = f"""fullname: {st.session_state["fullname"]}, 
                        age: {st.session_state["age"]}, 
                        nationality: {st.session_state["nationality"]}, 
                        university: {st.session_state["university"]}, 
                        graduate_level: {st.session_state["study_level"]}, 
                        major: {st.session_state["major"]}, 
                        funding-method: {st.session_state["funding_method"]}"""

    TEMPLATE_OFFICER = f"""{st.secrets['TEMPLATE_OFFICER_ONE']} Human Predefined Info:{user_details} \n {st.secrets['TEMPLATE_OFFICER_TWO']}"""


    st.info(
        """
        Start by greeting and introducing yourself
        """,
        icon="👋"
    )

    prompt_template_officer = PromptTemplate.from_template(template=TEMPLATE_OFFICER)

    index = st.session_state.pinecone.Index(st.session_state.Index_name) 

    st.markdown(
        """
        <style>
            .stChatMessage {
                flex-direction: row-reverse;
                text-align: right;
            }
            .st-emotion-cache-4oy321 {
                flex-direction: row;
                text-align: left;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )

    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if st.session_state["interview_type"] == "Beginner":
        st.session_state.user_message_limit = 4
    elif st.session_state["interview_type"] == "Intermediate":  
        st.session_state.user_message_limit = 7
    else:
        st.session_state.user_message_limit = 10

    if st.session_state.user_message_count < st.session_state.user_message_limit:
        if not st.session_state.feedback_shown:
            if prompt := st.chat_input("Write your message here...", max_chars=500):
                    st.session_state.messages.append({"role":"user", "content":prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    if st.session_state.user_message_count < (st.session_state.user_message_limit - 1):
                        with st.chat_message("assistant"):
                            qa_desc = vector_db_fetch(prompt)
                            st.session_state.current_qa = qa_desc
                            
                            response = memory_chain.invoke(prompt)
                            st.write(response)

                            if(st.session_state.tips_shown == True):
                                st.info(
                                    f"""
                                    Answer Tip/Example: {st.session_state.current_qa.split("answer guide is")[1]}
                                    """,
                                    icon="ℹ️"
                                )
                            #tts = gTTS(text=response, lang='en')
                            #tts.save("output.mp3")
                            #audio = AudioSegment.from_file("output.mp3", format="mp3")
                            #faster_audio = audio.speedup(playback_speed=1.2) 
                            #play(faster_audio)
                        st.session_state.messages.append({"role":"assistant", "content": response})
                    
                    elif st.session_state.user_message_count == (st.session_state.user_message_limit - 1):
                        with st.chat_message("assistant"):
                            closing_response = "That's all I needed for today. You can see your Feedback in the next page."
                            st.session_state.chat_memory.save_context(inputs={'input':prompt}, outputs={'output':closing_response})
                            st.write(closing_response)
                        st.session_state.messages.append({"role": "assistant", "content": closing_response})

                    st.session_state.user_message_count += 1

    if st.session_state.user_message_count >= st.session_state.user_message_limit:
        st.session_state.chat_complete = True
            
if st.session_state.chat_complete and not st.session_state.feedback_shown:
    if st.button("Get Feedback", on_click=show_feedback):
        st.write("Fetching feedback...")

if st.session_state.feedback_shown:
    st.subheader(st.session_state.interview_type + " Interview Feedback")

    prompt_template_feedback = PromptTemplate.from_template(template=st.secrets["TEMPLATE_FEEDBACK"])

    feedback_chain = prompt_template_feedback | st.session_state.feedback_chat_instance | StrOutputParser()

    if(st.session_state.feedback_generated == False):
        feedback_response = feedback_chain.invoke({'conversation_history':st.session_state.chat_memory.load_memory_variables({})["message_log"]})  
        st.session_state.feedback_generated = True
        st.write(feedback_response)


    if st.button("Restart Interview", type="primary"):
        st.session_state.setup_complete = False
        st.session_state.messages = []
        st.session_state.user_message_count = 0
        st.session_state.feedback_shown = False
        st.session_state.chat_complete = False
        streamlit_js_eval(js_expressions="parent.window.location.reload()")