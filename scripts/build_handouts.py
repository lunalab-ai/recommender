#!/usr/bin/env python3
"""Build lecture PDFs and separate student quiz handouts from public sources."""

from __future__ import annotations

import argparse
import base64
import html
import mimetypes
import re
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import yaml
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
CSS = """
@page { size: A4; margin: 16mm 15mm 18mm; }
body { font-family: 'Noto Sans CJK KR', 'Malgun Gothic', sans-serif;
       font-size: 10pt; line-height: 1.65; color: #172333; }
h1 { font-size: 21pt; } h2 { font-size: 15pt; margin-top: 1.5em; }
h3 { font-size: 12pt; } h1,h2,h3,h4 { break-after: avoid; }
p,li { orphans: 3; widows: 3; overflow-wrap: anywhere; }
table { width: 100%; border-collapse: collapse; font-size: 9pt; }
th,td { border: 1px solid #ccd5df; padding: 6px; overflow-wrap: anywhere; }
th { background: #edf2f7; } tr { break-inside: avoid; }
thead { display: table-header-group; }
img { display: block; max-width: 100%; max-height: 220mm; margin: 12px auto;
      break-inside: avoid; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #f3f5f7;
      padding: 10px; font-size: 8.5pt; break-inside: avoid; }
code { font-family: 'DejaVu Sans Mono', Consolas, monospace; }
blockquote { border-left: 3px solid #a4bbd4; padding-left: 12px; margin-left: 0; }
a { color: #185c99; text-decoration: none; }
.quiz h2 { font-size: 11.5pt; margin-top: .9em; }
.quiz p { margin: .55em 0; }
"""


def quiz_questions(text: str) -> list[str]:
    """Read the numbered checkpoint quiz, rejecting missing or malformed questions."""
    match = re.search(r"^## 점검 퀴즈\s*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not match:
        raise ValueError("Missing ## 점검 퀴즈 section")
    rows = re.findall(r"^(\d+)\. (.+)$", match[1], re.M)
    if not 5 <= len(rows) <= 10 or [int(n) for n, _ in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Quiz must contain 5–10 consecutively numbered questions")
    return [question.strip() for _, question in rows]


def quiz_markdown(page: Path, response: Path) -> str:
    text = page.read_text(encoding="utf-8")
    questions = quiz_questions(text)
    data = yaml.safe_load(response.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("session") != page.stem:
        raise ValueError(f"Quiz session mismatch: {response}")
    items = data.get("items", [])
    if not isinstance(items, list) or len(items) != len(questions):
        raise ValueError(f"Quiz answer count mismatch: {response}")
    parts = [
        text.splitlines()[0] + " · 퀴즈 답안",
        "> 딥러닝응용I(추천시스템) · 동덕여자대학교 · 유원상 교수 · 2026-2",
        "먼저 강의자료의 점검 퀴즈를 풀고 확인하세요. 서술형 문항은 같은 의미의 다른 표현도 가능합니다.",
        "이 답안은 해당 강의의 설명을 바탕으로 정리한 학습용 해설입니다.",
    ]
    for number, (question, item) in enumerate(zip(questions, items, strict=True), 1):
        if not isinstance(item, dict) or item.get("question") != question:
            raise ValueError(f"Quiz question {number} changed; review its answer: {response}")
        answer = item.get("answer")
        if not isinstance(answer, str) or not answer.strip() or "TODO" in answer:
            raise ValueError(f"Missing reviewed answer {number}: {response}")
        parts.extend([f"## {number}. {question}", answer.strip()])
    return "\n\n".join(parts) + "\n"


def render_html(text: str, source: Path, root: Path) -> str:
    """Embed local teaching images; resolve document links to public GitHub URLs."""
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = md.parse(text)
    for token in tokens:
        for child in token.children or []:
            if child.type == "image":
                target = child.attrGet("src") or ""
                if urlparse(target).scheme:
                    raise ValueError(f"PDF images must be local: {target}")
                asset = (source.parent / unquote(target)).resolve()
                asset.relative_to((root / "course/notion/assets").resolve())
                mime = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
                child.attrSet(
                    "src", f"data:{mime};base64," + base64.b64encode(asset.read_bytes()).decode()
                )
            elif child.type == "link_open":
                target = child.attrGet("href") or ""
                if target and not urlparse(target).scheme and not target.startswith("#"):
                    rel = (source.parent / unquote(target)).resolve().relative_to(root.resolve())
                    child.attrSet(
                        "href",
                        "https://github.com/lunalab-ai/recommender/blob/main/" + rel.as_posix(),
                    )
    body = md.renderer.render(tokens, md.options, {})
    return (
        '<!doctype html><html lang="ko"><meta charset="utf-8"><title>'
        + html.escape(text.splitlines()[0].lstrip("# "))
        + f'</title><style>{CSS}</style><body class="'
        + ("quiz" if "· 퀴즈 답안" in text.splitlines()[0] else "lecture")
        + f'">{body}</body></html>'
    )


def build(root: Path = ROOT, *, check_only: bool = False) -> list[Path]:
    pages = sorted((root / "course/notion/sessions").glob("w[0-9][0-9][ab]-*.md"))
    if not pages:
        raise ValueError("No session pages found")
    documents = []
    for page in pages:
        response = root / "course/quiz" / f"{page.stem}.yml"
        quiz = quiz_markdown(page, response)
        documents.append((page, page.read_text(encoding="utf-8"), quiz))
    if check_only:
        return pages
    from playwright.sync_api import sync_playwright

    output = root / "course/handouts"
    output.mkdir(parents=True, exist_ok=True)
    built = []
    index = [
        "# 강의 PDF와 퀴즈 답안",
        "강의 PDF는 Notion용 강의 원본과 같은 내용을 담습니다. 퀴즈 답안은 별도 자료입니다.",
        "| 수업 | 강의 PDF | 퀴즈 답안 PDF | 퀴즈 답안 텍스트 |",
        "|---|---|---|---|",
    ]
    with tempfile.TemporaryDirectory(prefix="course-handouts-") as tmp, sync_playwright() as pw:
        browser = pw.chromium.launch()
        tab = browser.new_page()
        tab.route("http://**/*", lambda route: route.abort())
        tab.route("https://**/*", lambda route: route.abort())
        for source, lecture, quiz in documents:
            for suffix, content in [("", lecture), ("-quiz", quiz)]:
                name = source.stem + suffix
                html_path = Path(tmp) / f"{name}.html"
                html_path.write_text(render_html(content, source, root), encoding="utf-8")
                tab.goto(html_path.as_uri())
                tab.evaluate("document.fonts.ready")
                if not tab.evaluate(
                    "Array.from(document.images).every(i => i.complete && i.naturalWidth > 0)"
                ):
                    raise ValueError(f"Image failed to render: {source}")
                destination = output / f"{name}.pdf"
                tab.pdf(
                    path=str(destination),
                    prefer_css_page_size=True,
                    print_background=True,
                    tagged=True,
                    outline=True,
                    display_header_footer=True,
                    header_template="<span></span>",
                    footer_template='<div style="font-size:8px;width:100%;text-align:center;color:#667">'
                    '<span class="pageNumber"></span> / <span class="totalPages"></span></div>',
                )
                built.append(destination)
            quiz_path = output / f"{source.stem}-quiz.md"
            quiz_path.write_text(quiz, encoding="utf-8", newline="\n")
            built.append(quiz_path)
            stem = source.stem
            title = lecture.splitlines()[0].lstrip("# ")
            index.append(
                f"| {title} | [다운로드]({stem}.pdf) | [답안 PDF]({stem}-quiz.pdf) | [답안 보기]({stem}-quiz.md) |"
            )
        browser.close()
    index_path = output / "README.md"
    index_path.write_text(
        "\n\n".join(index[:2]) + "\n\n" + "\n".join(index[2:]) + "\n", encoding="utf-8"
    )
    built.append(index_path)
    from build_course_hub import build as build_hub

    built.extend(build_hub(root))
    return built


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Validate quiz alignment without rendering PDFs"
    )
    args = parser.parse_args()
    for path in build(check_only=args.check):
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
