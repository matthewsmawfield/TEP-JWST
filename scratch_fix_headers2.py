import re

# 1. Fix 4_results.html
with open("site/components/4_results.html", "r") as f:
    text = f.read()

text = text.replace("<h3>3.3 UNCOVER DR4: Mass-sSFR and Mass-Age Correlations</h3>", 
                    "<h3>3.2 UNCOVER DR4: Mass-sSFR and Mass-Age Correlations</h3>")

with open("site/components/4_results.html", "w") as f:
    f.write(text)

# 2. Fix 5_discussion.html
with open("site/components/5_discussion.html", "r") as f:
    text = f.read()

text = text.replace("<h4>4.6.1 $\\Lambda$CDM Tension Quantification</h4>", 
                    "<h4>4.3.1 $\\Lambda$CDM Tension Quantification</h4>")
text = text.replace("<h4>4.13.1 Critical Test: The Mass-Dust Inversion</h4>", 
                    "<h4>4.6.1 Critical Test: The Mass-Dust Inversion</h4>")

with open("site/components/5_discussion.html", "w") as f:
    f.write(text)

# 3. Fix 8_appendix.html
with open("site/components/8_appendix.html", "r") as f:
    text = f.read()

# Replace empty heading "<h4></h4>" or similar, let's see how it looks.
# Wait, let's check what it looks like first.
