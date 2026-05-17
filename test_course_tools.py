"""Lightweight regression checks for the course-selection business tools."""

from __future__ import annotations

import json
from pathlib import Path

from course_tools import (
    drop_course_impl,
    enroll_course_impl,
    get_my_courses_impl,
    search_courses_impl,
)

ENROLLMENTS_FILE = Path(__file__).resolve().parent / "enrollments.json"
BASELINE = {
    "20230101": ["CS101", "MA201"],
    "20230102": [],
    "20230103": ["MA305"],
    "20230104": ["CS303", "PH110"],
    "20230105": ["CS101"],
}


def reset_enrollments() -> None:
    ENROLLMENTS_FILE.write_text(json.dumps(BASELINE, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    reset_enrollments()
    assert "数据结构" in search_courses_impl(department="计算机学院", credits=3)
    assert "选课成功" in enroll_course_impl("20230102", "数据结构")
    assert "数据结构" in get_my_courses_impl("20230102")
    assert "当前没有选" in drop_course_impl("20230102", "高等数学")
    assert "高等数学" in search_courses_impl(department="数学")
    reset_enrollments()
    print("course tool checks passed")


if __name__ == "__main__":
    main()
