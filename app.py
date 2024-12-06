from typing import Literal, List

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages.base import BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils.ocr_data import analyze_layout


from typing import cast

from utils.prompts import ROUTING_PROMPT
from model.model import get_model

import chainlit as cl

from agents.med_summarizer import MedSummarizer
from agents.langmed_translator import LangMedTranslator
from agents.chat_agent import ChatAgent
from agents.doc_recommend import DocRecommend
from agents.med_saver import MedSaver

user_profile = {}


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource(s)."""

    datasources: List[
        Literal[
            "MedSummarizer",
            "LangMedTranslator",
            "MedSaver",
            "ChatAgent",
            "DocRecommend"
        ]
    ] = Field(
        default=[],
        description="Given a user question, choose which agent(s) would be most relevant for answering their question",
    )

    user_query: str = Field(..., description="The original query from the user")

    confidence_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Confidence scores for each selected datasource",
    )

    class Config:
        schema_extra = {
            "example": {
                "datasources": ["MedSummarizer", "MedSaver"],
                "user_query": "What are the recommendation medicine for this disease?",
                "confidence_scores": {"MedSaver": 0.9, "MedSummarizer": 0.6},
            }
        }


def get_route_name(llm, route_prompt, query, route_query: BaseModel):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", route_prompt),
            ("human", "{input}"),
        ]
    )
    structured_llm = llm.with_structured_output(route_query)
    query_router = prompt | structured_llm
    route = query_router.invoke(query)
    return route


def get_route_datasource(user_query):
    llm = get_model()
    route_datasource = get_route_name(llm, ROUTING_PROMPT, user_query, RouteQuery)
    print("I am datasources...", route_datasource.datasources[0])
    if len(route_datasource.datasources) > 0:
        return route_datasource.datasources[0]
    else:
        return "ChatAgent"


def setup_agents(llm_model):
    med_summarizer = MedSummarizer(llm_model)
    cl.user_session.set(
        "med_summarizer",
        med_summarizer,
    )
    langmed_translator = LangMedTranslator(llm_model)
    cl.user_session.set(
        "langmed_translator",
        langmed_translator,
    )
    chat_agent = ChatAgent(llm_model)
    cl.user_session.set(
        "chat_agent",
        chat_agent,
    )
    med_saver = MedSaver(llm_model)
    cl.user_session.set(
        "med_saver",
        med_saver,
    )
    doc_recommend = DocRecommend(llm_model)
    cl.user_session.set(
        "doc_recommend",
        doc_recommend,
    )


@cl.on_chat_start
async def on_chat_start():
    llm = get_model()
    cl.user_session.set(
        "memory",
        ChatMessageHistory(),
    )
    cl.user_session.set(
        "llm",
        llm
    )
    chat_history = cl.user_session.get("memory")
    setup_agents(llm)
    startup_message = '''
    Welcome! 🌟 I'm here to assist you with all things health-related and guide you toward a healthier lifestyle.
    To get started, please provide the below asked informations so I can tailor my recommendations to your needs.
    '''
    msg = cl.Message(content=startup_message)
    await msg.send()
    name = await cl.AskUserMessage(content="What is your name?").send()
    gender = await cl.AskActionMessage(
        content="What is your gender?",
        actions=[
            cl.Action(name="male", value="male", label="Male"),
            cl.Action(name="female", value="female", label="Female"),
            cl.Action(name="other", value="other", label="Other"),
            cl.Action(name="prefer not to say", value="prefer not to say", label="Prefer Not to Say"),
        ]
    ).send()
    age = await cl.AskActionMessage(
        content="What is your age?",
        actions=[
            cl.Action(name="0-5", value="0-5yrs", label="0-5yrs"),
            cl.Action(name="5-10", value="5-10yrs", label="5-10yrs"),
            cl.Action(name="10-15", value="10-15yrs", label="10-15yrs"),
            cl.Action(name="15-20", value="15-20yrs", label="15-20yrs"),
            cl.Action(name="20-25", value="20-25yrs", label="20-25yrs"),
            cl.Action(name="25-35", value="25-35yrs", label="25-35yrs"),
            cl.Action(name="35-45", value="35-45yrs", label="35-45yrs"),
            cl.Action(name="45-50", value="45-50yrs", label="45-50yrs"),
            cl.Action(name="50-60", value="50-60yrs", label="50-60yrs"),
            cl.Action(name="Above 60", value="Above 60", label="Above 60"),
        ],
    ).send()

    # Ask for prescription and lab report
    prescription_uploaded = await cl.AskActionMessage(
        content="Do you have a prescription to upload?",
        actions=[
            cl.Action(name="yes", value="yes", label="Yes"),
            cl.Action(name="no", value="no", label="No"),
        ],
    ).send()
    prescription = None
    if prescription_uploaded and prescription_uploaded.get("value") == "yes":
        while prescription is None:
            prescription = await cl.AskFileMessage(
                content="Please upload your prescription here!",
                accept=["png", "jpg", "jpeg"]
            ).send()
        msg = cl.Message(content="Processing file uploaded...")
        await msg.send()
        prescription = prescription[0]
        prescription_analysis = analyze_layout(file_path=prescription.path)

    lab_report_uploaded = None
    if prescription_uploaded:
        lab_report_uploaded = await cl.AskActionMessage(
            content="Do you have a lab report to upload?",
            actions=[
                cl.Action(name="yes", value="yes", label="Yes"),
                cl.Action(name="no", value="no", label="No"),
            ],
        ).send()
    lab_report = None
    if lab_report_uploaded and lab_report_uploaded.get("value") == "yes":
        while lab_report is None:
            lab_report = await cl.AskFileMessage(
                content="Please upload your lab report here!",
                accept=["png", "jpg", "jpeg"]
            ).send()
        msg = cl.Message(content="Processing file uploaded...")
        await msg.send()
        lab_report = lab_report[0]
        lab_report_analysis = analyze_layout(file_path=lab_report.path)

    # Analyze uploaded files
    if age:
        user_profile['age'] = age.get("value")
        chat_history.add_message(BaseMessage(content=f"My age is {age['value']}", type="human"))
    if gender:
        user_profile['gender'] = gender.get("value")
        chat_history.add_message(BaseMessage(content=f"My gender is {gender['value']}", type="human"))

    if name:
        user_profile['name'] = name.get("output")
        await cl.Message(content=f"Thanks, {name['output']}! Your profile is ready.").send()
        chat_history.add_message(BaseMessage(content=f"My name is {name['output']}", type="human"))

    if lab_report:
        # await cl.Message(content=f"Preliminary analysis of your lab report: {lab_report_analysis}").send()
        cl.user_session.set("lab_report", lab_report_analysis)
        chat_history.add_message(BaseMessage(content=f"My lab report is: {lab_report_analysis}", type="human"))
    if prescription:
        # await cl.Message(content=f"Preliminary analysis of your prescription: {prescription_analysis}").send()
        chat_history.add_message(BaseMessage(content=f"My prescription is: {prescription_analysis}", type="human"))
        cl.user_session.set("prescription", prescription_analysis)

    message = await cl.Message(
        content="👋 **Welcome to HealthMate!**\n\nWhat would you like to do next?\n\n"
                "1️⃣ **Summarize Key Information**: Get a clear and concise overview of your prescription or lab report.\n"
                "2️⃣ **Find Affordable Alternatives**: Identify cost-effective alternatives to your prescribed medicines.\n"
                "3️⃣ **Translate Details**: Convert your prescription details into your preferred language.\n\n"
                "4️⃣ **Doctor Recommendation**: Get recommendations on type of doctors based on your health conditions.\n\n"
                "Your health, your way—let's get started! 🚀"
    ).send()
    print("message sent is:", message.content)
    chat_history.add_message(BaseMessage(content=message.content, type="ai"))


@cl.on_message
async def on_message(message: cl.Message):
    user_query = message.content
    msg = cl.Message(content="")
    lab_report = cl.user_session.get("lab_report")
    prescription = cl.user_session.get("prescription")
    agent_name = get_route_datasource(user_query)
    print("Agent name is.......", agent_name)
    message.content = f'''
        User Query - {user_query}\n
        Prescription - {prescription}\n
        Lab Report - {lab_report}
    '''
    if agent_name == 'MedSummarizer':
        cls: MedSummarizer = cl.user_session.get("med_summarizer")
    elif agent_name == 'MedSaver':
        cls: MedSaver = cl.user_session.get("med_saver")
    elif agent_name == 'DocRecommend':
        cls: DocRecommend = cl.user_session.get("doc_recommend")
    elif agent_name == 'LangMedTranslator':
        cls: LangMedTranslator = cl.user_session.get("langmed_translator")
    else:
        cls: ChatAgent = cl.user_session.get("chat_agent")
    agent_executor = cls.get_executor()
    chat_history = cl.user_session.get("memory")
    runnable = RunnableWithMessageHistory(
        agent_executor,
        lambda _: chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    agent = cast(RunnableWithMessageHistory, runnable)
    async for chunk in agent.astream(
        {"input": message.content},
        config=RunnableConfig(
            callbacks=[cl.LangchainCallbackHandler(stream_final_answer=True)],
            configurable={"session_id": cl.user_session.get("id")},
            ),
    ):
        if "output" in chunk:
            chat_history.add_message(
                BaseMessage(
                    content=str(user_query) + str(chunk["output"]),
                    type="ai",
                )
            )

            await msg.stream_token(
                chunk["output"] 
                + "<br>\n\n\n\n<small><i><b>Disclaimer</b>: The information provided by this chatbot is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a healthcare provider for medical concerns.</small></i>"
            )

    await msg.send()
