#!/usr/bin/env python3
"""Generate the student README and consolidated references from public course sources."""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import yaml
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
MD = MarkdownIt("commonmark").enable("table")


def cell(value: object) -> str:
    """Keep user-authored text inside one Markdown table cell."""
    return str(value).replace("|", "&#124;").replace("\n", " ")


def local_link(root: Path, label: str, relative: str) -> str:
    return f"[{label}]({relative})" if (root / relative).is_file() else "—"


def public_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError(f"Expected a public HTTPS URL: {value}")
    return value


def read_sessions(root: Path, metadata: dict) -> list[dict]:
    """Discover lessons, require explicit dates/refs, and never infer holiday dates."""
    lessons = []
    seen = set()
    for page in sorted((root / "course/notion/sessions").glob("w[0-9][0-9][ab]-*.md")):
        sid = page.stem[:4]
        if sid in seen:
            raise ValueError(f"Duplicate session page: {sid}")
        seen.add(sid)
        if sid not in metadata:
            raise ValueError(f"Register {sid} in course/hub.yml before building the hub")
        info = metadata[sid]
        raw_date = info.get("date")
        date = dt.date.fromisoformat(str(raw_date)) if raw_date else None
        ref = info.get("colab_ref", "main")
        if not isinstance(ref, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", ref):
            raise ValueError(f"Invalid Colab ref for {sid}")
        text = page.read_text(encoding="utf-8")
        heading = text.splitlines()[0]
        if not heading.startswith("# "):
            raise ValueError(f"Missing session title: {page}")
        title = heading[2:].split(" · ", 1)[-1]
        notion = info.get("notion_url")
        if notion:
            public_url(notion)
        lessons.append(
            dict(
                id=sid,
                stem=page.stem,
                page=page,
                title=title,
                text=text,
                date=date,
                ref=ref,
                notion=notion,
            )
        )
    return lessons


def session_table(root: Path, lessons: list[dict], repository: str) -> str:
    lines = [
        "| 차시 | 강의 날짜 | 강의 주제 | 강의노트 | 퀴즈 | 실습 ipynb | Colab |",
        "|---|---|---|---|---|---|---|",
    ]
    for lesson in lessons:
        sid, stem, ref = lesson["id"], lesson["stem"], lesson["ref"]
        date = lesson["date"]
        display_date = (
            f"{date.isoformat()} ({'월화수목금토일'[date.weekday()]})" if date else "일정 미정"
        )
        notes = [
            local_link(root, "MD", f"course/notion/sessions/{stem}.md"),
            local_link(root, "PDF", f"course/handouts/{stem}.pdf"),
        ]
        if lesson["notion"]:
            notes.append(f"[Notion]({lesson['notion']})")
        quiz = [
            local_link(root, "답안", f"course/handouts/{stem}-quiz.md"),
            local_link(root, "PDF", f"course/handouts/{stem}-quiz.pdf"),
        ]
        notebook = f"notebooks/student/{stem}.ipynb"
        if (root / notebook).is_file():
            suffix = f"{repository}/blob/{quote(ref, safe='')}/{notebook}"
            ipynb = f"[ipynb](https://github.com/{suffix})"
            label = "최신본" if ref == "main" else "수업본"
            colab = f"[{label}](https://colab.research.google.com/github/{suffix})"
        else:
            ipynb = colab = "—"
        lines.append(
            "| "
            + " | ".join(
                [
                    sid.upper(),
                    display_date,
                    cell(lesson["title"]),
                    " · ".join(notes),
                    " · ".join(quiz),
                    ipynb,
                    colab,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def resource_table(resources: list[dict], *, prefix: str = "") -> str:
    lines = ["| 분야 | 자료 | 언어 | 활용 방법 |", "|---|---|---|---|"]
    for item in resources:
        url = item["url"]
        if urlparse(url).scheme:
            public_url(url)
        else:
            url = prefix + url
        lines.append(
            f"| {cell(item['group'])} | [{cell(item['title'])}]({url}) | "
            f"{cell(item['language'])} | {cell(item['purpose'])} |"
        )
    return "\n".join(lines)


def references(
    lessons: list[dict], resources: list[dict], root: Path, *, checked_on: str = "미기록"
) -> str:
    parts = [
        "# 전체 참고자료",
        "[강의 허브](../README.md) · [Python 복습 길잡이](python-basics.md)",
        "## 공통 보충 자료",
        f"수업을 위해 추가한 보충 읽기이며 외부 원문은 각 링크에서 확인합니다. 공통 보충 링크 확인일: {checked_on}.",
        resource_table(resources, prefix="../"),
        "## 차시별 출처와 참고 링크",
        "각 강의의 참고자료 절과 본문 외부 링크를 모았습니다. 출처 구분 설명은 강의 원문을 유지합니다.",
    ]
    for lesson in lessons:
        parts.extend(
            [
                f"### {lesson['id'].upper()} · {lesson['title']}",
                f"[강의 원문](notion/sessions/{lesson['stem']}.md)",
            ]
        )
        match = re.search(r"^## [^\n]*참고[^\n]*\n(.*?)(?=^## |\Z)", lesson["text"], re.M | re.S)
        section = (
            match[1].strip() if match else "강의 원문에 참고자료 절이 아직 작성되지 않았습니다."
        )

        # Rebase relative Markdown links when copying the source reference section.
        def rebase(match: re.Match, page: Path = lesson["page"]) -> str:
            target = match[2]
            if urlparse(target).scheme or target.startswith("#"):
                return match[0]
            resolved = (page.parent / unquote(target)).resolve()
            relative = resolved.relative_to(root.resolve()).as_posix()
            return f"{match[1]}(../{relative})"

        section = re.sub(r"(!?\[[^\]]*\])\(([^)]+)\)", rebase, section)
        parts.append(section)
        links = {}
        for token in MD.parse(lesson["text"]):
            children = token.children or []
            for i, child in enumerate(children):
                if child.type == "link_open":
                    url = child.attrGet("href") or ""
                    if url.startswith("https://") and url not in section:
                        label_parts = []
                        for part in children[i + 1 :]:
                            if part.type == "link_close":
                                break
                            label_parts.append(part.content)
                        label = "".join(label_parts)
                        links.setdefault(url, label or url)
        if links:
            parts.extend(
                [
                    "본문에서 함께 소개한 링크:",
                    "\n".join(f"- [{label}]({url})" for url, label in links.items()),
                ]
            )
    return "\n\n".join(parts) + "\n"


def validate_links(text: str, path: Path, root: Path) -> None:
    """Fail on broken or escaping local links, including links in Markdown tables."""
    for token in MD.parse(text):
        for child in token.children or []:
            if child.type not in {"link_open", "image"}:
                continue
            target = child.attrGet("href" if child.type == "link_open" else "src") or ""
            if urlparse(target).scheme:
                public_url(target)
                continue
            if target.startswith("#"):
                continue
            resolved = (path.parent / unquote(target.split("#")[0])).resolve()
            resolved.relative_to(root.resolve())
            if not resolved.exists():
                raise ValueError(f"Broken local link in {path.name}: {target}")


def build(root: Path = ROOT, *, check: bool = False) -> list[Path]:
    config = yaml.safe_load((root / "course/course.yml").read_text(encoding="utf-8"))
    hub = yaml.safe_load((root / "course/hub.yml").read_text(encoding="utf-8"))
    repository = hub["repository"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Invalid public repository name")
    lessons = read_sessions(root, hub["sessions"])
    course = config["course"]
    resources = hub["resources"]
    textbook = course["textbook"]
    assessment = course["assessment"]
    days = {"Tuesday": "화", "Thursday": "목", "Monday": "월", "Wednesday": "수", "Friday": "금"}
    values = {
        "TITLE": course["korean_title"],
        "IDENTITY": f"{course['institution']} {course['department']} · {course['semester']} · {course['credits']}학점\n\n담당교수 **{course['instructor']}** · {'·'.join(days[d] for d in course['meeting']['days'])} {course['meeting']['time']}",
        "SESSIONS": session_table(root, lessons, repository),
        "OUTCOMES": "\n".join(f"- {item}" for item in course["outcomes"]),
        "CURRICULUM": "| 주차 | 주제 | 학습목표 |\n|---|---|---|\n"
        + "\n".join(
            f"| {week['week']} | {cell(' · '.join(week['topics']))} | {cell(week['goal'])} |"
            for week in config["weeks"]
        ),
        "PYTHON_RESOURCES": resource_table([r for r in resources if r["group"] == "Python 기초"]),
        "TEXTBOOK": f"주교재: {textbook['author']}, 『{textbook['title']}』, {textbook['publisher']}, {textbook['year']}.",
        "ASSESSMENT": f"중간고사 **{assessment['midterm']}%** · 기말고사 **{assessment['final']}%** · 과제물 **{assessment['assignments']}%** · 출석 **{assessment['attendance']}%**\n\n실습 과제는 원칙적으로 통과 {course['assignment_policy']['pass_points']}점 또는 탈락 {course['assignment_policy']['fail_points']}점으로 평가합니다.",
    }
    readme = (root / "course/readme-template.md").read_text(encoding="utf-8")
    for key, value in values.items():
        readme = readme.replace("{{" + key + "}}", value)
    if re.search(r"\{\{\w+\}\}", readme):
        raise ValueError("Unresolved README template field")
    outputs = {
        root / "README.md": readme,
        root / "course/references.md": references(
            lessons, resources, root, checked_on=str(hub.get("resources_checked", "미기록"))
        ),
    }
    # Generated pages can link to each other before either exists on the first build.
    for path, text in outputs.items():
        if check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                raise ValueError(f"Stale hub output: {path}; run scripts/build_course_hub.py")
        else:
            path.write_text(text, encoding="utf-8", newline="\n")
    for path, text in outputs.items():
        validate_links(text, path, root)
    validate_links(
        (root / "course/python-basics.md").read_text(encoding="utf-8"),
        root / "course/python-basics.md",
        root,
    )
    return list(outputs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for path in build(check=args.check):
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
