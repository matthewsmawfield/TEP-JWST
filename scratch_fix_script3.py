with open("scripts/steps/step_160_manuscript_consistency_check.py", "r") as f:
    text = f.read()

text = text.replace('"<h3>3.2 UNCOVER DR4: Mass-sSFR and Mass-Age Correlations</h3>",\n', "")
text = text.replace('"results_3_3_uncover": "<h3>3.3 UNCOVER DR4: Mass-sSFR and Mass-Age Correlations</h3>"', 
                    '"results_3_2_uncover": "<h3>3.2 UNCOVER DR4: Mass-sSFR and Mass-Age Correlations</h3>"')

with open("scripts/steps/step_160_manuscript_consistency_check.py", "w") as f:
    f.write(text)
