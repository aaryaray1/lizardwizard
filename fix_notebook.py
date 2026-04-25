import json

# Read the notebook
with open('03_Model_Evaluation_and_Testing.ipynb', 'r') as f:
    nb = json.load(f)

# Cells to convert from markdown to code (those with actual Python code, not section headers)
# Based on structure: section headers stay as markdown
# All other code-containing cells convert to code
code_cell_indices = [4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 17, 18, 20, 21, 22, 24, 25, 27, 29, 30, 32, 33]

converted = 0
for idx in code_cell_indices:
    if idx < len(nb['cells']):
        if nb['cells'][idx]['cell_type'] == 'markdown':
            nb['cells'][idx]['cell_type'] = 'code'
            # Add execution_count and outputs if not present
            if 'execution_count' not in nb['cells'][idx]:
                nb['cells'][idx]['execution_count'] = None
            if 'outputs' not in nb['cells'][idx]:
                nb['cells'][idx]['outputs'] = []
            converted += 1

# Write the modified notebook
with open('03_Model_Evaluation_and_Testing.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("✓ Converted markdown cells to code cells")
print(f"✓ Successfully converted {converted} cells to code type")

# Verify
with open('03_Model_Evaluation_and_Testing.ipynb', 'r') as f:
    nb = json.load(f)

cells_by_type = {}
for i, c in enumerate(nb['cells']):
    ct = c['cell_type']
    if ct not in cells_by_type:
        cells_by_type[ct] = []
    cells_by_type[ct].append(i)

print(f"\nVerification - Final cell types:")
for cell_type in ['markdown', 'code']:
    if cell_type in cells_by_type:
        count = len(cells_by_type[cell_type])
        print(f"  {cell_type}: {count} cells")

print(f"\n✓ Notebook is ready to run!")
