# Save this as clean_tex.py and run: python3 clean_tex.py
import re
with open("paper.tex", "r") as f: content = f.read()
content = re.sub(r'\', '', content)
with open("paper.tex", "w") as f: f.write(content)