
import os

file_path = '/Users/matthewsmawfield/www/TEP-JWST/site/components/5_discussion.html'

with open(file_path, 'r') as f:
    content = f.read()

# Target 1: Falsification Analysis
target1_old = 'However, the "Double-Control" test (§3.5) confirms that the redshift-dependent component of TEP ($\sqrt{1+z}$) is detected at $> 5\sigma$ independent of mass, resolving this concern.'
target1_new = 'However, the "Mass-Control" test (§3.5) confirms that the redshift-dependent component of TEP ($\sqrt{1+z}$) is detected at $> 4\\sigma$ independent of mass, resolving this concern.'

# Target 2: Critical Caveat
# Note: I am copying the exact string from the read_file output to ensure matching.
# I noticed in previous read output that "both" was italicized as <em>both</em> but in the last grep it wasn't finding "Double-Control" which is weird.
# Let's try to match flexible on the italicized part or just identifying unique substrings.
# Actually, the last read_file output (lines 290-310) showed:
# <p>Critical caveat: When controlling for <em>stellar mass</em> ...
# Wait, I might have partially edited it or previous edits partially succeeded?
# Let's check the file content again in the script to be safe.

# In the last read_file (lines 296):
# <p>Critical caveat: When controlling for <em>stellar mass</em> (using proper log-residualization of the exponential $\Gamma_t$), only the dust correlation survives with the predicted positive sign ($\rho = +0.28$). The age ratio correlation becomes <em>negative</em> ($\rho = -0.07$). However, the discovery of Resolved Core Screening provides a mass-independent test: the gradient depends on potential depth <em>profile</em>, not total mass, and its inversion in massive systems strongly favors the screening model.</p>

# It seems "both" was already removed or I am misremembering the original state vs current state.
# Ah, I see in the previous tool outputs that I tried to replace "controlling for *both* redshift and stellar mass" with "controlling for *stellar mass*".
# And the read_file output shows "controlling for <em>stellar mass</em>". So that part might have been updated or I am reading the wrong version.
# Wait, looking at the `read_file` output from the previous turn (lines 290-310 of 5_discussion.html):
# 296:    <p>Critical caveat: When controlling for <em>stellar mass</em> (using proper log-residualization of the exponential $\Gamma_t$), only the dust correlation survives with the predicted positive sign ($\rho = +0.28$). The age ratio correlation becomes <em>negative</em> ($\rho = -0.07$). ...

# So "both" is gone. The current state is "controlling for stellar mass".
# But the rest of the sentence is still old: "The age ratio correlation becomes negative...".
# I want to change it to "the age ratio and metallicity correlations vanish...".

target2_old_part = 'only the dust correlation survives with the predicted positive sign ($\rho = +0.28$). The age ratio correlation becomes <em>negative</em> ($\rho = -0.07$).'
target2_new_part = 'the age ratio and metallicity correlations vanish ($\\rho \\approx 0$), indicating they are largely driven by mass scaling in the general population. However, the <strong>$z > 8$ dust correlation</strong> survives with the predicted positive sign ($\\rho = +0.28$, $p < 10^{-5}$).'

# Check if targets exist
if target1_old not in content:
    print("Target 1 not found exactly.")
    # Try to find the sentence to replace anyway
    if 'the "Double-Control" test' in content:
        print("Found 'Double-Control' test string, attempting robust replace.")
        # We'll construct a regex or just replace the substring if unique
        content = content.replace('the "Double-Control" test', 'the "Mass-Control" test')
        content = content.replace('> 5\sigma', '> 4\sigma')
        print("Replaced Target 1 components.")
    else:
        print("Could not find Target 1 components.")

if target2_old_part not in content:
    print("Target 2 not found exactly.")
    print("Content around 'Critical caveat':")
    start = content.find("Critical caveat")
    if start != -1:
        print(content[start:start+300])
else:
    content = content.replace(target2_old_part, target2_new_part)
    print("Replaced Target 2.")

with open(file_path, 'w') as f:
    f.write(content)

print("File write complete.")
