"""Generate an offline, reproducible experiment summary from a saved JSON run.

Usage: python make_report.py results/experiment_YYYYMMDD_HHMMSS.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python make_report.py results/experiment_*.json')
    source = Path(sys.argv[1])
    data = json.loads(source.read_text(encoding='utf-8'))
    output = source.parent / 'results/published/summary.md'
    keys = [('baseline_train', 'Baseline / TRAIN'),
            ('selected_train', 'Selected / TRAIN'),
            ('baseline_holdout', 'Baseline / HOLDOUT'),
            ('selected_holdout', 'Selected / HOLDOUT')]
    rows = []
    for key, label in keys:
        entry = data[key]
        results = entry.get('results', [])
        api_errors = sum(any(token in str(r.get('answer', '')) for token in (
            'ServerError:', '500 INTERNAL', '429 RESOURCE_EXHAUSTED', 'ReadTimeout')) for r in results)
        rows.append((label, round(entry['accuracy'] * 100, 1), round(entry['tool_trace_compliance'] * 100, 1), api_errors))
    text = ['# AgentForge — экспериментальный отчёт', '',
            f"Модель: `{data.get('model', 'unknown')}`. Выбор кандидата основан только на TRAIN.", '',
            '| Версия / набор | Answer accuracy | Tool-trace compliance | Сбои API |',
            '|---|---:|---:|---:|']
    for label, accuracy, compliance, errors in rows:
        text.append(f'| {label} | {accuracy:g}% | {compliance:g}% | {errors} |')
    text += ['', '## Интерпретация',
             'Оценки на небольших выборках — предварительные. Изменение на TRAIN само по себе не доказывает обобщающего улучшения.',
             'Если были ошибки API, доля успешных задач включает инфраструктурные сбои и не является чистой метрикой рассуждения.',
             'Если оба результата на HOLDOUT одинаковы, нельзя заявлять об улучшении на контрольном наборе.',
             'Набор HOLDOUT не использовался при выборе инструкции.', '']
    output.write_text('\n'.join(text), encoding='utf-8')
    width, height = 760, 320
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="22" y="32" font-size="20" font-family="Arial">AgentForge: accuracy by split</text>']
    colors = ['#778899','#2c7a7b','#778899','#2c7a7b']
    for i, (label, accuracy, compliance, errors) in enumerate(rows):
        y = 68 + i * 61
        parts.append(f'<text x="22" y="{y+17}" font-size="13" font-family="Arial">{escape(label)}</text>')
        parts.append(f'<rect x="220" y="{y}" width="470" height="25" fill="#eef0f2" rx="4"/>')
        parts.append(f'<rect x="220" y="{y}" width="{470 * accuracy / 100:.2f}" height="25" fill="{colors[i]}" rx="4"/>')
        parts.append(f'<text x="700" y="{y+17}" font-size="14" font-family="Arial">{accuracy:g}%</text>')
    parts.append('</svg>')
    (source.parent / 'results/published/accuracy.svg').write_text('\n'.join(parts), encoding='utf-8')
    print('Created', output)
    print('Created', source.parent / 'results/published/accuracy.svg')


if __name__ == '__main__':
    main()
