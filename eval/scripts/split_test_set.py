from __future__ import annotations 

import argparse 
import json 
import random 
from collections import defaultdict
from pathlib import Path 
from typing import Any 

DEFAULT_SPLIT = {
    'text': (18, 7), 
    'table': (18, 7), 
    'image': (18, 7), 
    'mixed': (10, 5), 
    'unanswerable': (6, 4), 
}

def load_jsonl(path:Path) -> list[dict[str, Any]]: 
    rows = []
    with path.open('r', encoding='utf-8') as file: 
        for line_number, line in enumerate(file, start=1): 
            line=line.strip()
            if not line: 
                continue
            try: 
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc: 
                raise ValueError(
                    f'JSON invalid at {line_number}: {exc}'
                ) from exc
    return rows

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None: 
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file: 
        for row in rows: 
            file.write(json.dumps(row, ensure_ascii=False) + '\n')

def split_by_category(
    rows: list[dict[str, Any]], 
    seed: int, 
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: 
        category = row.get('category')
        if category not in DEFAULT_SPLIT: 
            raise ValueError(f'Category invalid: {category}')
        grouped[category].append(row)

    random_generator = random.Random(seed)
    development = []
    test = []
    for category, (development_count, test_count) in DEFAULT_SPLIT.items(): 
        category_rows = grouped[category]
        expected_count = development_count + test_count 

        if len(category_rows) != expected_count: 
            raise ValueError(
                f"Category '{category}' cần {expected_count} câu, "
                f"nhưng tìm thấy {len(category_rows)} câu."
            )
        
        random_generator.shuffle(category_rows)
        development.extend(category_rows[:development_count])
        test.extend(category_rows[development_count:])
    random_generator.shuffle(development)
    random_generator.shuffle(test)

    return development, test 

def validate_split(development: list[dict[str, Any]], test: list[dict[str, Any]],) -> None: 
    development_ids = {row['id'] for row in development}
    test_ids = {row['id'] for row in test}

    duplicated_ids = development_ids & test_ids

    if duplicated_ids: 
        raise ValueError(f'Development & test duplicated id: {sorted(duplicated_ids)}')
    
    if len(development) != 70: 
        raise ValueError(f'Development must have 70 query, now {len(development)}')
    if len(test) != 30: 
        raise ValueError(f'Test must have 30 query, now {len(test)}')

def main() -> None: 
    parser = argparse.ArgumentParser(
        description='Devise evaluation dataset to development and test set'
    )
    parser.add_argument(
        '--input', 
        type=Path, 
        default=Path('eval/test_sets/fpt_2025_qa_100.jsonl')
    )
    parser.add_argument(
        '--development-output', 
        type=Path, 
        default=Path("eval/test_sets/fpt_2025_dev.jsonl"),
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=Path("eval/test_sets/fpt_2025_test.jsonl"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed giúp kết quả chia có thể tái lập.",
    )

    args = parser.parse_args()

    rows = load_jsonl(args.input)

    if len(rows) != 100:
        raise ValueError(f"Dataset phải có 100 câu, hiện có {len(rows)}")

    development, test = split_by_category(rows, args.seed)
    validate_split(development, test)

    write_jsonl(args.development_output, development)
    write_jsonl(args.test_output, test)

    print(f"Development: {len(development)} câu")
    print(f"Test: {len(test)} câu")
    print(f"Development file: {args.development_output}")
    print(f"Test file: {args.test_output}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()
        




