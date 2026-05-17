# 基于 LangChain Agent 的智能选课助手

本项目实现了一个“自然语言选课助手”。学生可以直接用中文完成查课、选课、退课、查看课表等操作，Agent 会通过 LangChain Tool Calling 自动选择并调用工具。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `courses.json` | 课程库，包含 6 门课程，字段包括课程号、课程名、教师、院系、学分、容量、已选人数、上课时间。 |
| `enrollments.json` | 学生选课记录，包含 5 个学生的初始选课数据。 |
| `course_tools.py` | 纯业务逻辑：加载 JSON、查询课程、选课、退课、联表查看课表，并把选课/退课结果写回 `enrollments.json`。 |
| `course_agent.py` | LangChain Agent 装配：使用 `@tool` 封装 4 个工具，并通过 `create_tool_calling_agent` + `AgentExecutor` 串联工具调用。 |
| `test_course_tools.py` | 不依赖真实 LLM 的业务回归检查脚本。 |
| `requirements.txt` | 运行 LangChain Agent 需要的 Python 依赖。 |

## 实现的工具

### `search_courses`

按课程名、老师、院系、学分查询课程：

- `name`：课程名称关键词，模糊匹配。
- `teacher`：教师姓名关键词，模糊匹配。
- `department`：院系关键词，模糊匹配。
- `credits`：学分，精确匹配。

至少需要一个查询条件。返回课程号、课程名、教师、院系、学分、时间和余量。

### `enroll_course`

为学生选课，会检查：

1. 课程是否存在。
2. 课程是否满员。
3. 学生是否重复选课。

选课成功后会写回 `enrollments.json`。

### `drop_course`

为学生退课，会检查该学生是否确实已经选择该课程。退课成功后会写回 `enrollments.json`。

### `get_my_courses`

查看学生当前已选课程。该工具会读取 `enrollments.json` 中的课程号，并联表查询 `courses.json` 中的课程详情。

## 运行方式

安装依赖：

```bash
python -m pip install -r requirements.txt
```

运行作业要求的 5 条测试用例（使用同一套 `AgentExecutor`，并开启 `verbose=True`）：

```bash
python course_agent.py --demo
```

使用真实 LLM 进入交互模式：

```bash
export OPENAI_API_KEY="你的 API Key"
python course_agent.py --interactive
```

业务逻辑回归检查：

```bash
python test_course_tools.py
```

## 测试用例与 verbose=True 日志

> 下面日志展示 `AgentExecutor(verbose=True)` 的关键过程。实际终端输出可能因 LangChain 版本和模型措辞略有不同，但工具调用和工具返回应一致。

### 1. “帮我查一下计算机学院有哪些 3 学分的课”

```text
> Entering new AgentExecutor chain...
Invoking: `search_courses` with `{'department': '计算机学院', 'credits': 3}`

找到 3 门课程：
CS101｜数据结构｜教师：张伟｜院系：计算机学院｜学分：3｜时间：周一 1-2 节｜余量：15
CS202｜人工智能导论｜教师：王强｜院系：计算机学院｜学分：3｜时间：周三 5-6 节｜余量：0
CS303｜数据库系统｜教师：陈晨｜院系：计算机学院｜学分：3｜时间：周五 3-4 节｜余量：15
我查到了这些可参考的课程：
找到 3 门课程：
CS101｜数据结构｜教师：张伟｜院系：计算机学院｜学分：3｜时间：周一 1-2 节｜余量：15
CS202｜人工智能导论｜教师：王强｜院系：计算机学院｜学分：3｜时间：周三 5-6 节｜余量：0
CS303｜数据库系统｜教师：陈晨｜院系：计算机学院｜学分：3｜时间：周五 3-4 节｜余量：15
> Finished chain.
```

### 2. “我是 20230102，把数据结构选上”

```text
> Entering new AgentExecutor chain...
Invoking: `enroll_course` with `{'student_id': '20230102', 'course_id_or_name': '数据结构'}`

选课成功：学生 20230102 已选上 CS101（数据结构），上课时间：周一 1-2 节。
选课成功：学生 20230102 已选上 CS101（数据结构），上课时间：周一 1-2 节。
> Finished chain.
```

### 3. “我（20230102）现在的课表是什么？”

```text
> Entering new AgentExecutor chain...
Invoking: `get_my_courses` with `{'student_id': '20230102'}`

学生 20230102 当前已选 1 门课程：
CS101｜数据结构｜教师：张伟｜院系：计算机学院｜学分：3｜时间：周一 1-2 节｜余量：15
学生 20230102 当前已选 1 门课程：
CS101｜数据结构｜教师：张伟｜院系：计算机学院｜学分：3｜时间：周一 1-2 节｜余量：15
> Finished chain.
```

### 4. “高等数学不想上了，帮我退掉”

```text
> Entering new AgentExecutor chain...
Invoking: `drop_course` with `{'student_id': '20230102', 'course_id_or_name': '高等数学'}`

退课失败：学生 20230102 当前没有选 MA201（高等数学），不能退未选课程。
退课失败：学生 20230102 当前没有选 MA201（高等数学），不能退未选课程。
> Finished chain.
```

### 5. “我想再选一门数学类的课，有什么推荐？”

```text
> Entering new AgentExecutor chain...
Invoking: `search_courses` with `{'department': '数学'}`

找到 2 门课程：
MA201｜高等数学｜教师：李娜｜院系：数学学院｜学分：4｜时间：周二 3-4 节｜余量：12
MA305｜概率论与数理统计｜教师：赵敏｜院系：数学学院｜学分：3｜时间：周四 1-2 节｜余量：20
我查到了这些可参考的课程：
找到 2 门课程：
MA201｜高等数学｜教师：李娜｜院系：数学学院｜学分：4｜时间：周二 3-4 节｜余量：12
MA305｜概率论与数理统计｜教师：赵敏｜院系：数学学院｜学分：3｜时间：周四 1-2 节｜余量：20
> Finished chain.
```

## 设计要点

- 工具 docstring 明确说明用途、参数和返回值，方便 LLM 判断何时调用工具以及如何传参。
- JSON 读写集中在 `course_tools.py`，便于测试和维护。
- 工具函数捕获业务异常并返回中文友好提示，避免把 Python 异常堆栈暴露给 Agent。
- `--demo` 模式提供确定性演示模型，方便在没有 API Key 时复现 Tool Calling 流程；`--interactive` 模式可接入真实 ChatOpenAI 模型。
