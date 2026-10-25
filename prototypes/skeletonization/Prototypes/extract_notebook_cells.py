import json
import re

notebook_path = "keyPointsExtraction.ipynb"
script_path = "run_iteration9_extracted.py"

print(f"Reading notebook {notebook_path}...")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb.get("cells", [])
extracting = False
code_lines = []

for cell in cells:
    cell_type = cell.get("cell_type", "")
    source = cell.get("source", [])
    
    if cell_type == "markdown":
        text = "".join(source).strip()
        if "## iteration 9  implementation" in text:
            extracting = True
            print("Found Iteration 9 Implementation section. Starting extraction...")
            continue
            
    if extracting and cell_type == "code":
        # Join lines of code
        code = "".join(source)
        # Skip empty cells
        if not code.strip():
            continue
        code_lines.append(code)

print(f"Extracted {len(code_lines)} code cells.")

# Write the compiled python script
with open(script_path, "w", encoding="utf-8") as out:
    out.write("\n\n# --- EXTRACTED CELL ---\n\n".join(code_lines))

print(f"Extracted script written to {script_path}.")
