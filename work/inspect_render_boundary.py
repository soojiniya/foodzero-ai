from pathlib import Path

p = Path("src/foodzero/asos_mapping.py")
text = p.read_text(encoding="utf-8")
start = text.index("def render_review_doc")
end = text.index('\n\nif __name__ == "__main__":', start)
print(repr(text[start : start + 500]))
print("---END---")
print(repr(text[end : end + 80]))
