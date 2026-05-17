"""Business logic for the LangChain course-selection agent.

The functions in this module are intentionally framework-agnostic so they can be
unit-tested without an LLM. LangChain ``@tool`` wrappers live in ``course_agent.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
COURSES_FILE = BASE_DIR / "courses.json"
ENROLLMENTS_FILE = BASE_DIR / "enrollments.json"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def load_courses() -> list[dict[str, Any]]:
    """Load all courses from the JSON course catalog."""
    return _load_json(COURSES_FILE, [])


def load_enrollments() -> dict[str, list[str]]:
    """Load all student enrollment records from JSON storage."""
    return _load_json(ENROLLMENTS_FILE, {})


def save_enrollments(enrollments: dict[str, list[str]]) -> None:
    """Persist student enrollment records to JSON storage."""
    _write_json(ENROLLMENTS_FILE, enrollments)


def _course_to_line(course: dict[str, Any]) -> str:
    available = int(course["capacity"]) - int(course["enrolled"])
    return (
        f'{course["course_id"]}｜{course["name"]}｜教师：{course["teacher"]}｜'
        f'院系：{course["department"]}｜学分：{course["credits"]}｜'
        f'时间：{course["time"]}｜余量：{available}'
    )


def _find_course(course_id_or_name: str) -> dict[str, Any] | None:
    keyword = course_id_or_name.strip().lower()
    for course in load_courses():
        if course["course_id"].lower() == keyword or course["name"].lower() == keyword:
            return course
    return None


def search_courses_impl(
    name: str = "",
    teacher: str = "",
    department: str = "",
    credits: int | None = None,
) -> str:
    """Search courses using fuzzy text filters and an optional exact credit filter."""
    try:
        if not any([name, teacher, department, credits is not None]):
            return "请至少提供一个查询条件：课程名、老师、院系或学分。"

        courses = load_courses()
        results: list[dict[str, Any]] = []
        for course in courses:
            if name and name.strip().lower() not in course["name"].lower():
                continue
            if teacher and teacher.strip().lower() not in course["teacher"].lower():
                continue
            if department and department.strip().lower() not in course["department"].lower():
                continue
            if credits is not None and int(course["credits"]) != int(credits):
                continue
            results.append(course)

        if not results:
            return "没有找到符合条件的课程。你可以放宽课程名、老师、院系或学分条件后再试。"

        lines = [f"找到 {len(results)} 门课程："]
        lines.extend(_course_to_line(course) for course in results)
        return "\n".join(lines)
    except Exception as exc:
        return f"查询课程时遇到问题：{exc}。请检查查询条件后重试。"


def enroll_course_impl(student_id: str, course_id_or_name: str) -> str:
    """Enroll a student in a course and persist the updated enrollment record."""
    try:
        student_id = student_id.strip()
        course = _find_course(course_id_or_name)
        if course is None:
            return f"选课失败：没有找到课程“{course_id_or_name}”。"

        enrollments = load_enrollments()
        selected_courses = enrollments.setdefault(student_id, [])
        course_id = course["course_id"]
        if course_id in selected_courses:
            return f"选课失败：学生 {student_id} 已经选过 {course_id}（{course['name']}），不能重复选课。"
        if int(course["enrolled"]) >= int(course["capacity"]):
            return f"选课失败：{course_id}（{course['name']}）已满员，当前容量 {course['capacity']} 人。"

        selected_courses.append(course_id)
        save_enrollments(enrollments)
        return f"选课成功：学生 {student_id} 已选上 {course_id}（{course['name']}），上课时间：{course['time']}。"
    except Exception as exc:
        return f"选课时遇到问题：{exc}。请稍后重试或联系教务老师。"


def drop_course_impl(student_id: str, course_id_or_name: str) -> str:
    """Drop a course for a student and persist the updated enrollment record."""
    try:
        student_id = student_id.strip()
        course = _find_course(course_id_or_name)
        if course is None:
            return f"退课失败：没有找到课程“{course_id_or_name}”。"

        enrollments = load_enrollments()
        selected_courses = enrollments.get(student_id, [])
        course_id = course["course_id"]
        if course_id not in selected_courses:
            return f"退课失败：学生 {student_id} 当前没有选 {course_id}（{course['name']}），不能退未选课程。"

        selected_courses.remove(course_id)
        enrollments[student_id] = selected_courses
        save_enrollments(enrollments)
        return f"退课成功：学生 {student_id} 已退掉 {course_id}（{course['name']}）。"
    except Exception as exc:
        return f"退课时遇到问题：{exc}。请稍后重试或联系教务老师。"


def get_my_courses_impl(student_id: str) -> str:
    """Return detailed course schedule information for one student."""
    try:
        student_id = student_id.strip()
        enrollments = load_enrollments()
        selected_ids = enrollments.get(student_id, [])
        if not selected_ids:
            return f"学生 {student_id} 当前没有已选课程。"

        courses_by_id = {course["course_id"]: course for course in load_courses()}
        lines = [f"学生 {student_id} 当前已选 {len(selected_ids)} 门课程："]
        for course_id in selected_ids:
            course = courses_by_id.get(course_id)
            if course is None:
                lines.append(f"{course_id}｜课程库中未找到该课程详情，请联系教务老师核对。")
            else:
                lines.append(_course_to_line(course))
        return "\n".join(lines)
    except Exception as exc:
        return f"查看课表时遇到问题：{exc}。请稍后重试或联系教务老师。"
