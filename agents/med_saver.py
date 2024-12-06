"""Identifies affordable alternatives to the prescribed medicines without compromising treatment effectiveness."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor, tool
from utils.prompts import MEDSAVER_PROMPT


class MedSaver:
    """Agent to perform actions related to Medical Recommendation."""

    def __init__(self, llm) -> None:
        self.llm = llm

    @tool
    def get_word_length(word: str) -> int:
        """LangMed Translator

        Args:
            word (str): _description_

        Returns:
            int: _description_
        """
        return len(word)

    def get_executor(self):
        tools = [self.get_word_length]
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", MEDSAVER_PROMPT),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        llm_with_tools = self.llm.bind_tools(tools)

        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
            }
            | prompt
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )

        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        return agent_executor
