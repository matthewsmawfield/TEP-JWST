with open("site/components/8_appendix.html", "r") as f:
    text = f.read()

# Replace multi-line h4 with single-line h4
import re
text = re.sub(r"<h4>\s*(B\.3\.3[^<]+)\s*</h4>", lambda m: "<h4>" + " ".join(m.group(1).split()) + "</h4>", text)

with open("site/components/8_appendix.html", "w") as f:
    f.write(text)
