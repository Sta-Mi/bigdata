"""LangChain ReAct-style tool-calling agent for university course selection.

Run examples:
    python course_agent.py --demo
    python course_agent.py --interactive

The interactive mode requires an OpenAI-compatible API key configured for
``langchain_openai.ChatOpenAI``. The demo mode uses the same AgentExecutor path
with a deterministic chat model so the assignment examples can be reproduced.
"""

from __future__ import annotations

import argparse
from typing import Any, Sequence

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, tool
from pydantic import Field, PrivateAttr

from course_tools import (
    drop_course_impl,
    enroll_course_impl,
    get_my_courses_impl,
    search_courses_impl,
)


@tool
def search_courses(
    name: str = "",
    teacher: str = "",
    department: str = "",
    credits: int | None = None,
) -> str:
    """按课程名、老师、院系、学分查询课程。

    用途：当用户想“查课”“找课”“推荐某类课程”时调用。本工具支持多条件组合查询，
    其中课程名、老师、院系均为模糊匹配，学分为精确匹配。至少需要提供一个参数；
    如果用户说“数学类”，通常把 department 设为“数学”，如果用户说“计算机学院”，
    把 department 设为“计算机学院”。

    参数：
        name: 课程名称关键词，例如“数据结构”；不知道课程名时传空字符串。
        teacher: 教师姓名关键词，例如“张伟”；不知道教师时传空字符串。
        department: 院系关键词，例如“计算机学院”或“数学”；不知道院系时传空字符串。
        credits: 学分数字，例如 3；用户没有指定学分时传 None。

    返回：中文字符串，包含匹配课程的课程号、课程名、教师、院系、学分、上课时间、余量；
    若没有结果或参数不足，返回友好的中文提示。
    """
    return search_courses_impl(name=name, teacher=teacher, department=department, credits=credits)


@tool
def enroll_course(student_id: str, course_id_or_name: str) -> str:
    """为指定学生选课。

    用途：当用户明确表示要“选课”“把某门课选上”“报名某课程”时调用。调用前需要知道
    student_id 和课程号或准确课程名；如果用户只给了模糊描述，应先调用 search_courses 查清楚。
    本工具会检查课程是否存在、课程是否满员、该学生是否已经选过该课，并在成功后把变化
    持久化写回 enrollments.json。

    参数：
        student_id: 学号字符串，例如“20230102”。
        course_id_or_name: 课程号或课程名，例如“CS101”或“数据结构”。

    返回：中文字符串，说明选课成功或失败原因；不会返回 Python 异常堆栈。
    """
    return enroll_course_impl(student_id=student_id, course_id_or_name=course_id_or_name)


@tool
def drop_course(student_id: str, course_id_or_name: str) -> str:
    """为指定学生退课。

    用途：当用户表示“不想上”“退掉”“取消某门课”时调用。调用前需要知道 student_id 和
    课程号或准确课程名；如果课程描述不清，应先调用 search_courses 查清楚。本工具会检查
    学生是否真的选了该课程，并在成功后把变化持久化写回 enrollments.json。

    参数：
        student_id: 学号字符串，例如“20230102”。
        course_id_or_name: 课程号或课程名，例如“MA201”或“高等数学”。

    返回：中文字符串，说明退课成功或失败原因；不会返回 Python 异常堆栈。
    """
    return drop_course_impl(student_id=student_id, course_id_or_name=course_id_or_name)


@tool
def get_my_courses(student_id: str) -> str:
    """查看指定学生当前所有已选课程及课程详情。

    用途：当用户询问“我的课表”“我选了哪些课”“当前课程安排”时调用。该工具会读取
    enrollments.json 中该学生的课程号，再关联 courses.json 的课程库详情。

    参数：
        student_id: 学号字符串，例如“20230102”。

    返回：中文字符串，列出该学生所有已选课程的课程号、课程名、教师、院系、学分、时间、余量；
    如果没有已选课程，返回友好的中文提示。
    """
    return get_my_courses_impl(student_id=student_id)


TOOLS: list[BaseTool] = [search_courses, enroll_course, drop_course, get_my_courses]

SYSTEM_PROMPT = """你是某大学教务处的智能选课助手。你能帮助学生查课、选课、退课和查看课表。
请遵循以下规则：
1. 尽量用工具获取真实课程和选课数据，不要编造课程。
2. 选课和退课必须调用对应工具完成持久化。
3. 如果缺少学号或课程信息，先追问；如果课程描述模糊，先查课再操作。
4. 回复要简洁、友好，并用中文说明操作结果。
"""


def build_prompt() -> ChatPromptTemplate:
    """Build the prompt required by create_tool_calling_agent."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )


def build_agent_executor(llm: BaseChatModel, verbose: bool = True) -> AgentExecutor:
    """Create the LangChain tool-calling agent and wrap it in AgentExecutor."""
    agent = create_tool_calling_agent(llm=llm, tools=TOOLS, prompt=build_prompt())
    return AgentExecutor(agent=agent, tools=TOOLS, verbose=verbose, handle_parsing_errors=True)


class DemoCourseChatModel(BaseChatModel):
    """Small deterministic chat model used only for reproducible local demos."""

    _bound_tools: Sequence[Any] = PrivateAttr(default_factory=list)
    _step: int = PrivateAttr(default=0)
    student_id: str = Field(default="20230102")

    @property
    def _llm_type(self) -> str:
        return "demo-course-chat-model"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "DemoCourseChatModel":
        clone = self.model_copy()
        clone._bound_tools = tools
        return clone

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        last = messages[-1]
        if getattr(last, "type", "") == "tool":
            content = str(last.content)
            final = self._final_answer(content)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=final))])

        human_text = next((str(message.content) for message in reversed(messages) if message.type == "human"), "")
        tool_name, args = self._decide_tool(human_text)
        self._step += 1
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": f"demo_call_{self._step}",
                    "name": tool_name,
                    "args": args,
                }
            ],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _decide_tool(self, text: str) -> tuple[str, dict[str, Any]]:
        if "计算机学院" in text and "3 学分" in text:
            return "search_courses", {"department": "计算机学院", "credits": 3}
        if "数据结构" in text and ("选" in text or "选上" in text):
            return "enroll_course", {"student_id": self.student_id, "course_id_or_name": "数据结构"}
        if "课表" in text or "现在的课" in text:
            return "get_my_courses", {"student_id": self.student_id}
        if "高等数学" in text and ("退" in text or "不想上" in text):
            return "drop_course", {"student_id": self.student_id, "course_id_or_name": "高等数学"}
        if "数学类" in text or "推荐" in text:
            return "search_courses", {"department": "数学"}
        return "search_courses", {"name": text}

    def _final_answer(self, tool_content: str) -> str:
        if tool_content.startswith("找到"):
            return f"我查到了这些可参考的课程：\n{tool_content}"
        return tool_content


def get_default_llm() -> BaseChatModel:
    """Create the default production LLM for interactive use."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def run_demo() -> None:
    """Run the five assignment examples with verbose AgentExecutor logs."""
    questions = [
        "帮我查一下计算机学院有哪些 3 学分的课",
        "我是 20230102，把数据结构选上",
        "我（20230102）现在的课表是什么？",
        "高等数学不想上了，帮我退掉",
        "我想再选一门数学类的课，有什么推荐？",
    ]
    executor = build_agent_executor(DemoCourseChatModel(), verbose=True)
    chat_history: list[BaseMessage] = []
    for index, question in enumerate(questions, start=1):
        print(f"\n===== 用例 {index}: {question} =====")
        result = executor.invoke({"input": question, "chat_history": chat_history})
        print(f"最终回答：{result['output']}")
        chat_history.extend([HumanMessage(content=question), AIMessage(content=result["output"])])


def run_interactive() -> None:
    """Start an interactive command-line assistant with the production LLM."""
    executor = build_agent_executor(get_default_llm(), verbose=True)
    chat_history: list[BaseMessage] = []
    print("智能选课助手已启动，输入 exit 退出。")
    while True:
        user_input = input("学生：").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            break
        if not user_input:
            continue
        result = executor.invoke({"input": user_input, "chat_history": chat_history})
        print(f"助手：{result['output']}")
        chat_history.extend([HumanMessage(content=user_input), AIMessage(content=result["output"])])


def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain 智能选课助手")
    parser.add_argument("--demo", action="store_true", help="运行作业要求的 5 条测试用例")
    parser.add_argument("--interactive", action="store_true", help="使用真实 LLM 进入交互模式")
    args = parser.parse_args()

    if args.interactive:
        run_interactive()
    else:
        run_demo()


if __name__ == "__main__":
    main()
