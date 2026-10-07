"""直接读取本机Word原生段落与公式；只记录本轮需要的短证据，不公开原件。"""

import json
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
path = ROOT / 'handoff/protocol/LOCAL_v2_0.docx'
with ZipFile(path) as source:
    body = ET.fromstring(source.read('word/document.xml')).find('w:body', NS)
paragraphs = [''.join(p.itertext()) for p in body.findall('w:p', NS)]
for i, p in enumerate(paragraphs):
    if any(token in p for token in ('25.3', '27.', 'L2', 'L3', '修复', '学习率')):
        print(f'PARAGRAPH {i}: {p}')
tables = body.findall('w:tbl', NS)
for i, table in enumerate(tables):
    text = ''.join(table.itertext())
    if i < 100 and any(token in text for token in ('(44)', '(47)', '(48)', '(49)', '(53)', '(54)',
                                     '（44）', '（47）', '（48）', '（49）', '（53）', '（54）')):
        print(f'NATIVE TABLE {i}: {text}')
    elif 'Stage 2' in text or ('L1' in text and 'L2' in text):
        print(f'STAGE / REPAIR TABLE {i}: {text}')
native = json.loads((ROOT / 'handoff/protocol/LOCAL_v2_NATIVE_FORMULAS.json').read_text(
    encoding='utf-8'))
print('NATIVE RECORD STRUCTURE:', list(native) if isinstance(native, dict) else len(native))
