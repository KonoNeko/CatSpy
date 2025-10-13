"""
Simple CSV -> JSONL converter for training data.
Usage:
  python csv_to_jsonl.py --input /path/to/dataset_phishing.csv --output ../data/merged_train.jsonl --text-col text --label-col label --label-map phishing:1,safe:0

The script will try to detect columns named 'text' and 'label' if not provided.
"""
import argparse
import csv
import json
from pathlib import Path


def parse_label_map(s: str):
    m = {}
    if not s:
        return m
    for pair in s.split(','):
        if ':' in pair:
            k,v = pair.split(':',1)
            m[k.strip()] = int(v.strip())
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--text-col', default='text')
    parser.add_argument('--label-col', default='label')
    parser.add_argument('--label-map', default='phishing:1,legitimate:0',
                        help="Comma-separated mapping for textual labels, e.g. 'phishing:1,safe:0'."
                        )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    assert in_path.exists(), f"Input not found: {in_path}"

    label_map = parse_label_map(args.label_map)

    with in_path.open('r', encoding='utf-8', errors='ignore') as fin, out_path.open('w', encoding='utf-8') as fout:
        reader = csv.DictReader(fin)
        # if text/label not present, try first two columns
        header = reader.fieldnames or []
        # Try the provided text/label columns, fall back to common names
        text_col = args.text_col if args.text_col in header else (header[0] if header else None)
        # label column: prefer explicit arg, then 'label', then 'status'
        if args.label_col in header:
            label_col = args.label_col
        elif 'label' in header:
            label_col = 'label'
        elif 'status' in header:
            label_col = 'status'
        elif len(header) > 1:
            label_col = header[1]
        else:
            label_col = None
        if text_col is None or label_col is None:
            raise ValueError('Could not determine text/label columns. Please pass --text-col and --label-col')

        count = 0
        for row in reader:
            text = row.get(text_col, '').strip()
            raw_label = row.get(label_col, '').strip()
            # normalize raw label to lower/strip for mapping
            raw_label_norm = raw_label.strip().lower()
            if raw_label_norm in label_map:
                label = label_map[raw_label_norm]
            else:
                try:
                    label = int(float(raw_label))
                except Exception:
                    # fallback: treat empty or unknown as 0
                    label = 0
            obj = {'text': text, 'label': label}
            fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
            count += 1
    print(f'Wrote {count} records to {out_path}')


if __name__ == '__main__':
    main()
