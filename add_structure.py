#!/usr/bin/env python3
"""Add structure markdown cells to pi_cde.ipynb"""
import json
import uuid

# Load notebook
with open('notebooks/pi_cde.ipynb', 'r') as f:
    nb = json.load(f)

cells = nb['cells']

# Define sections to insert (position, content)
# Positions refer to indices where to INSERT (cell will be inserted BEFORE this index)
sections = [
    (2, "## Section 1: Data Generation\n### 1-D Periodic Transport Equation\n\nIn this section we generate synthetic data for the 1-D transport equation:\n$$\\frac{\\partial u}{\\partial t} + c \\, \\frac{\\partial u}{\\partial x} = 0$$\n\nWe create multiple random initial conditions and evolve them analytically, producing snapshot matrices with known derivatives."),
    (4, "### 1.1 Domain Setup & Analytical Solution\n\nGenerate periodic Fourier series initial conditions and solve the transport equation analytically using interpolation. Compute derivatives using high-order finite differences."),
    (6, "### 1.2 Visualization of Generated Data\n\nVisualize the snapshot matrices and sensor readings to confirm data quality and periodicity."),
    (9, "## Section 2: Neural CDE Solver\n### Data Preparation & Model Setup\n\nBuild the neural CDE pipeline:\n1. Select sensor locations from the spatial grid\n2. Construct lag-windowed sequences of sensor readings\n3. Extract POD basis from training snapshots\n4. Project targets into POD coefficient space\n5. Initialize and train the NeuralCDE model"),
]

# Insert sections in REVERSE order so indices don't shift
for pos, content in sorted(sections, key=lambda x: x[0], reverse=True):
    cell = {
        "cell_type": "markdown",
        "metadata": {"id": str(uuid.uuid4())[:12]},
        "source": [content]
    }
    cells.insert(pos, cell)

# Save
with open('notebooks/pi_cde.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Added {len(sections)} section headers. New cell count: {len(cells)}")
