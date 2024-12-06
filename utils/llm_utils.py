from langchain.chains.llm import LLMChain
from langchain_core.prompts import (
    PromptTemplate,
)

from ai_telemedicine.utils.prompts import LAB_REPORT_PROMPT_TEMPLATE


def summarize_lab_report(user_query, report, llm):
    prompt = PromptTemplate(input_variables=["report", "user_query"], template=LAB_REPORT_PROMPT_TEMPLATE)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    output = llm_chain.apply(input_list=[{"report": f"{report}", "user-query": f"{user_query}"}])[0].get(
        "text"
    )
    return output
