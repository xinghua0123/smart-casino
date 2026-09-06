from pathlib import Path
code = Path(__file__).resolve().parents[1] / 'analytics.py'
exec(compile(code.read_text(), str(code), 'exec'))
