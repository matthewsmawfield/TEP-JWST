import glob
from pathlib import Path

def build():
    components = sorted(glob.glob("site/components/*.html"))
    output = []
    
    # Simple concatenation
    for comp in components:
        with open(comp, 'r') as f:
            output.append(f.read())
            
    # Write to dist/index.html
    out_dir = Path("site/dist")
    out_dir.mkdir(exist_ok=True)
    
    with open("site/dist/index.html", "w") as f:
        f.write("\n".join(output))

if __name__ == "__main__":
    build()
