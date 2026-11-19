# Lab Q.4 – ML + Quantum + Parallel Computing

**Notebook:** `OT_LabQ4_241070044.ipynb`  |  **Marks:** 5  |  **Random seed:** 44

Choosing the best 5 of 12 image-feature groups for a crop-leaf classifier, using a neural network, a QUBO, a quantum algorithm (VQE) and parallel computing.

---

## 1. What was asked

A crop-image classifier uses **12 feature groups** (colour, texture, shape, ...). To fit the memory limit of an edge device, only **5 groups** may be kept.

The plan in the question:
1. A small MLP classifies leaves as healthy / unhealthy using synthetic data. Its predictions show how good or bad each feature group is.
2. A quantum algorithm picks **exactly 5 of the 12 groups** (12 yes/no choices).
3. Parallel computing speeds up the repeated runs.

| Part | Task | Marks |
|---|---|---|
| **a. Machine Learning** | Make about 600 synthetic rows (seed 44), split 70/30, train a small MLP (scikit-learn). Report accuracy, precision, recall. Show the value for each of the 12 feature groups. | 1 (CO1) |
| **b. QUBO** | Write the energy function, build the Q matrix from the predictions, add a penalty for breaking the "exactly 5" rule, and check with brute force that the lowest-energy answer is correct. | 1 (CO2) |
| **c. Quantum Computing** | Solve the QUBO with VQE (hardware-efficient ansatz) on a simulator. Report the best answer, its energy and how often it appears. Compare with brute force. | 1.5 (CO2) |
| **d. Parallel Computing** | Run a grid of starting points in parallel with `joblib.Parallel`. Report time with 1, 2 and 4 workers, the speed-up and the efficiency. | 1 (CO4) |
| **e. Conclusion** | 3 lines: final choice and what was learned. | – |

---

## 2. What we did

### (a) Machine Learning
- Made **600 synthetic rows** with 24 numbers each. Every 2 numbers form 1 feature group, giving **12 groups**.
  - Groups 0–3: truly informative.
  - Groups 4–6: copies or mixes of the informative ones (redundant).
  - Groups 7–11: mostly noise.
- Split the data **70% train / 30% test**, then scaled it.
- Trained an **MLP with one hidden layer of 16 neurons**.
- To find the **value of each group**, we shuffled that group in the test data and measured how much accuracy dropped (average of 30 shuffles). A big drop means a useful group.

### (b) QUBO
Energy function (low energy = good choice):

```
E(x) = − Σ sᵢ·xᵢ  +  λ · Σ rᵢⱼ·xᵢ·xⱼ  +  P · (Σ xᵢ − 5)²
```

- `xᵢ` = 1 if group *i* is kept, else 0.
- `sᵢ` = usefulness of group *i* (from part a, scaled to 0–1).
- `rᵢⱼ` = how similar groups *i* and *j* are (absolute correlation). Keeping similar groups costs extra.
- `P(Σxᵢ − 5)²` = the **penalty**. It is zero only when exactly 5 groups are kept.
- Values used: `λ = 0.15`, `P = 2.0`.

Since xᵢ² = xᵢ, this becomes one matrix:
- `Q[i][i] = −sᵢ + P(1 − 2·5)`
- `Q[i][j] = λ·rᵢⱼ + 2P` (for i < j)
- constant = `P·5²`

Then `E(x) = xᵀQx + constant`. We tried **all 4096** possible answers (brute force) to find the true lowest energy.

### (c) Quantum Computing (VQE)
- Simulated **12 qubits** with plain NumPy (no real quantum hardware).
- **Hardware-efficient ansatz:** 4 layers of RY rotations on every qubit, with CZ gates in a ring between layers.
- A classical optimizer (**COBYLA**, up to 1500 steps) adjusts the angles to lower the average energy.
- Ran VQE from **8 random starting points** and kept the best one.
- Measured the final state **4096 times (shots)** to see which answer appears most.

### (d) Parallel Computing
- Ran **16 different starting points** of VQE (400 steps each) with `joblib.Parallel`.
- Timed it with **1, 2 and 4 workers**.
- **Speed-up** = time with 1 worker ÷ time with n workers. **Efficiency** = speed-up ÷ n.

---

## 3. Results

### (a) MLP performance (test set)

| Accuracy | Precision | Recall |
|---|---|---|
| **0.861** | **0.822** | **0.922** |

Value of each feature group (accuracy drop when the group is shuffled):

| Group | Name | Value |
|---|---|---|
| x0 | colour_mean | 0.1007 |
| x1 | colour_hist | 0.0665 |
| x2 | texture_glcm | **0.1054** |
| x3 | texture_lbp | 0.0704 |
| x4 | shape_area | 0.0296 |
| x5 | shape_perim | 0.0039 |
| x6 | edge_density | 0.0491 |
| x7 | vein_pattern | 0.0000 |
| x8 | spot_count | 0.0000 |
| x9 | leaf_ratio | 0.0000 |
| x10 | brightness | 0.0000 |
| x11 | noise_level | 0.0017 |

The informative groups (colour and texture) score highest. The noise groups score about 0, as expected.

### (b) QUBO and brute force (4096 states)

| Item | Result |
|---|---|
| Lowest energy | **−3.4781** |
| Best bit-string | `111100100000` |
| Groups chosen | colour_mean, colour_hist, texture_glcm, texture_lbp, edge_density |
| Exactly 5 groups? | Yes |
| Best "keep 5" answer = global best? | Yes (the penalty works) |

### (c) VQE

Energies from the 8 random starts: −2.03, −2.96, −2.94, −2.48, −2.94, −1.61, −2.67, −2.94.

| Item | Result |
|---|---|
| Best VQE bit-string | `101100100010` |
| Groups chosen | colour_mean, texture_glcm, texture_lbp, edge_density, brightness |
| Energy of that state | **−2.9583** |
| Final VQE expectation energy | −2.9555 |
| How often it appeared | **4069 / 4096 shots (99.3%)** |
| Matches brute force? | **No** |

**Comparison with brute force**

| | Bit-string | Energy |
|---|---|---|
| Brute force (true best) | `111100100000` | −3.4781 |
| VQE (best found) | `101100100010` | −2.9583 |

VQE gave a **valid answer (exactly 5 groups)** and picked 4 of the same 5 groups as brute force. It chose `brightness` instead of `colour_hist`. The energy is about 0.52 higher than the true best, so VQE got stuck in a **local minimum**.

### (d) Parallel timing (16 starting points)

| Workers | Time (s) | Speed-up | Efficiency |
|---|---|---|---|
| 1 | 55.88 | 1.00× | 1.00 |
| 2 | 48.58 | 1.15× | 0.58 |
| 4 | 51.20 | 1.09× | 0.27 |

The best energy over the grid was −2.7731 for all three runs, so parallel running does not change the answer, only the time. The speed-up is small, and 4 workers were not better than 2. This is likely because of the extra cost of starting worker processes and sharing work, and because of how many CPU cores the machine has.

---

## 4. Conclusion

1. **Final choice (brute force, true best):** colour_mean, colour_hist, texture_glcm, texture_lbp and edge_density. VQE found a valid, close answer (`101100100010`, energy −2.958) but not the exact best (−3.478).
2. The **penalty term** makes "exactly 5" work in the QUBO, but it also makes the VQE search harder, so many random restarts are needed.
3. **Parallel runs** helped only a little (1.15× with 2 workers, efficiency 0.58); more workers do not always mean faster.

---

## 5. How to run

```bash
pip install numpy scipy scikit-learn joblib
jupyter notebook OT_LabQ4_241070044.ipynb
```

Run all cells from top to bottom. VQE and the parallel step take a few minutes. Timings will differ on other computers, but the ML scores and the brute-force answer stay the same because the seed is fixed.
