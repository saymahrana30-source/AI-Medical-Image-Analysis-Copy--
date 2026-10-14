# 5. Virtual simulation

You have no hospital access, so you **simulate the workflow** with public data. Be clear in the README
that it is a simulation.

## How the simulation maps to reality
| Real hospital | This project |
|---|---|
| Patient chest X-ray arrives | Test image drawn at random |
| Radiologist-confirmed diagnosis | Folder label (NORMAL / PNEUMONIA) - hidden from the model |
| PACS / worklist | `worklist.csv` |
| AI triage software | `src/simulate.py` + the trained model |
| Radiologist review | You inspect flagged cases, Grad-CAM, and the misses |

The model never sees test images during training, so its predictions on them behave like "new patients".
The simulation is **not** a clinical trial: one public dataset, no workflow, no real consequences.

## Steps
1. Train and evaluate: `python main.py train` then `python main.py evaluate`.
2. Run the queue: `python main.py simulate --patients 60`.
3. Patients arrive in random order (FIFO queue).
4. For each, the model returns P(pneumonia); inference time is measured on your machine.
5. Triage tiers: **URGENT** (P >= 0.85), **REVIEW** (P >= threshold), **ROUTINE** (below).
6. The worklist is re-sorted by probability; the script compares the average queue position of true
   pneumonia cases in arrival order vs AI order.
7. Look at the top cases with Grad-CAM, and at every missed case (false negative).

## Outputs (in `outputs/simulation/`)
`worklist.csv`, `simulation_summary.json`, `simulation_dashboard.png`, `simulation_gradcam_top_cases.png`.

## Proof to capture
Dashboard PNG, worklist CSV excerpt, Grad-CAM grid, terminal screenshot of the summary, and a short
paragraph explaining what the simulation does and does not show. Quote only numbers from your own run.
