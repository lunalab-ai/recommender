"""Opt-in board-review layout shared with the registered course PDF builder."""
from pathlib import Path
import re
import yaml

MODE = 'board-review-pdf'
CSS = '''
@page{size:A4;margin:15mm 18mm 16mm}*{box-sizing:border-box}
body{font-size:11pt;line-height:1.4;color:#20242a}h1{font-size:22pt}h2{font-size:14pt}h3{font-size:10.5pt}
p{margin:2.1mm 0}li{margin:1.2mm 0}pre{font-size:9.8pt;line-height:1.35;padding:2.5mm;break-inside:avoid}
table{font-size:10pt;line-height:1.35;margin:3mm 0;break-inside:avoid}th,td{padding:1.6mm 2mm}
.pdf-question{break-before:page;break-inside:auto}.board-reflection{break-before:page}
.board-writing-space{height:80mm;break-inside:avoid;border:1px solid #9da5ae;margin-top:4mm;position:relative;background:white}
.board-writing-label{position:absolute;top:1.5mm;left:2.5mm;padding:0 1mm;background:white;font-size:9pt;color:#4d5661}
'''

def settings_for_session(root: Path, sid: str):
    """Read explicit layout metadata; ordinary lessons keep their existing renderer."""
    path = root / 'course/hub.yml'
    if not path.is_file():
        return None
    hub = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    item = hub.get('sessions', {}).get(sid, {})
    mode = item.get('material_mode')
    if not mode:
        return None
    if mode != MODE:
        raise ValueError('Unknown lesson material_mode: ' + str(mode))
    count = item.get('board_review_questions')
    if type(count) is not int or not 1 <= count <= 99:
        raise ValueError('Board review requires an explicit question count from 1 to 99')
    return item

def settings_for_page(root: Path, source: Path):
    """Apply the registered mode only to that session's canonical public note."""
    relative = source.resolve().relative_to(root.resolve()).as_posix()
    if not relative.startswith('course/notion/sessions/'):
        return None
    return settings_for_session(root, source.stem.split('-', 1)[0])

def render_body(body: str, *, question_count: int) -> str:
    """Insert one 80mm writing box per question without accepting source HTML."""
    ids = [int(x) for x in re.findall(r'<h2>Q(\d{2})\s*·', body)]
    if ids != list(range(1, question_count + 1)):
        raise ValueError('Board review questions must appear exactly once in source order')
    label = '<h3>판서·계산 메모</h3>'
    def question(match):
        fragment = match.group(1)
        if fragment.count(label) != 1:
            raise ValueError('Each board review question requires one writing heading')
        qid = re.search(r'<h2>(Q\d{2})', fragment).group(1)
        box = ('<div class="board-writing-space" data-board-question="' + qid + '">'
               '<span class="board-writing-label">' + qid + ' · 판서·계산 메모</span></div>')
        return '<section class="pdf-question">' + fragment.replace(label, box) + '</section>'
    body = re.sub(r'(<h2>Q\d{2}\s*·[\s\S]*?)(?=<h2>|$)', question, body)
    return body.replace('<h2>수업 후 복습 기록</h2>', '<h2 class="board-reflection">수업 후 복습 기록</h2>')
