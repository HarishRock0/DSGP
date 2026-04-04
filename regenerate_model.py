#!/usr/bin/env python
"""
Quick model regeneration script.
Directly loads and executes the notebook to regenerate the model pickle.
"""

import sys
import os
import subprocess

print("\n" + "="*70)
print("🔄 REGENERATING SKILLDEV MODEL FROM NOTEBOOK")
print("="*70)

# Get the DSGP directory (parent of the script)
dsgp_dir = os.path.dirname(os.path.abspath(__file__))
ml_dir = os.path.join(dsgp_dir, 'ML')

# Run jupyter nbconvert to execute the notebook
notebook_path = os.path.join(ml_dir, 'skilldev.ipynb')
output_path = os.path.join(ml_dir, 'skilldev_output.ipynb')

print(f"\n📖 Executing notebook: {notebook_path}")
cmd = [
    sys.executable, '-m', 'jupyter', 'nbconvert',
    '--to', 'notebook',
    '--execute',
    '--ExecutePreprocessor.timeout=300',
    f'--output={output_path}',
    notebook_path
]

try:
    # Change to ML directory so relative paths in notebook work correctly
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=ml_dir)
    if result.returncode == 0:
        print(f"✅ Notebook executed successfully!")
        print(f"✅ Model should be saved to: {os.path.join(dsgp_dir, 'model', 'skilldev_model.pkl')}")
    else:
        print(f"❌ Notebook execution failed:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)
except subprocess.TimeoutExpired:
    print(f"❌ Notebook execution timed out after 600 seconds")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
