"""Recommend the doctor on the basis of diseases."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor, tool
from utils.prompts import DOC_RECOMMEND_PROMPT


class DocRecommend:
    """Agent to perform actions related to doctor recommendation."""

    def __init__(self, llm) -> None:
        self.llm = llm

    @tool
    def get_word_length(word: str) -> int:
        """DocRecommend Agent

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
                ("system", DOC_RECOMMEND_PROMPT),
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
