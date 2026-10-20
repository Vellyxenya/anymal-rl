# Controlled gait evaluation: before vs. after four-foot fine-tuning

## Question

Did the 500-iteration four-foot gait fine-tuning make the learned 1 m/s forward gait more trot-like and less asymmetric?

## Why these metrics

Quadruped gait is conventionally described using foot-contact timing: duty factor (the fraction of a stride spent in stance) and normalized phase relationships between limbs. A symmetric gait alternates homologous left/right limbs by half a cycle; a trot additionally synchronizes diagonal pairs. These descriptors follow the Hildebrand-style analysis used in comparative quadruped locomotion research.

There is no accepted universal scalar for "naturalness." This evaluation therefore reports the established descriptors separately. It also supplies a project-specific **Trot Deviation Index (TDI)** for experiment ranking:

`TDI = phase MAE + 0.25 × duty-factor CV + 0.25 × stride-period CV`

Lower is better. The index is explicitly not a standard biological metric and must not replace its components.

References:

- Usherwood & Self Davies, *Work minimization accounts for footfall phasing in slow quadrupedal gaits* (2017): duty factor and normalized limb phase definitions. https://pmc.ncbi.nlm.nih.gov/articles/PMC5599235/
- Patrick et al., *Minimalist analogue robot discovers animal-like walking gaits* (2020): symmetric gait, stance/swing, and trot phase relationships. https://pmc.ncbi.nlm.nih.gov/articles/PMC7655146/
- Dietz et al., *Interlimb Coordination in Human Crawling Reveals Similarities in Development and Neural Control With Quadrupeds* (2009): half-cycle homologous alternation and trot-like coordination. https://pmc.ncbi.nlm.nih.gov/articles/PMC2657078/

## Controlled protocol

| Setting | Value |
|---|---:|
| Before checkpoint | `model_2497.pt` |
| After checkpoint | `model_2996.pt` |
| Terrain | Deterministic infinite flat plane |
| Command | `(vx, vy, yaw) = (1.0 m/s, 0, 0)` |
| External pushes | Disabled |
| Command resampling | Disabled |
| Trials | 64 paired parallel environments |
| Seed | 42 for both policies |
| Warm-up | 2 s, discarded |
| Measurement window | 10 s |
| Contact threshold | 1 N |
| Trot phase targets | LF–RH = 0 cycles; LF–RF = 0.5; LF–LH = 0.5 |
| Uncertainty | Paired bootstrap, 20,000 resamples |

Both policies completed all 64 trials without falling.

## Results

| Metric | Before | After | Change | Interpretation |
|---|---:|---:|---:|---|
| Trot Deviation Index ↓ | 0.0751 | 0.0767 | **+2.1%** | Slightly worse overall |
| Touchdown phase MAE ↓ | 0.0264 cycles | 0.0391 cycles | **+48.4%** | Worse phase precision |
| Duty-factor CV ↓ | 0.1675 | 0.1212 | **−27.7%** | Substantially better four-foot balance |
| Duty-factor range ↓ | 0.1901 | 0.1826 | **−3.9%** | Slightly better balance |
| Stride-period CV ↓ | 0.0274 | 0.0290 | **+5.9%** | Slightly less regular |
| LF–RH contact agreement ↑ | 70.7% | 90.0% | **+19.3 points** | Much better diagonal synchronization |
| RF–LH contact agreement ↑ | 85.5% | 81.7% | **−3.8 points** | Worse for the other diagonal |
| Diagonal-pair opposition ↑ | 84.3% | 98.0% | **+13.7 points** | Much clearer alternating trot structure |
| Mean forward velocity | 1.007 m/s | 1.024 m/s | +0.017 m/s | Similar realized speed |
| Velocity tracking MAE ↓ | 0.0349 m/s | 0.0382 m/s | **+9.6%** | Slightly worse tracking |

Every listed paired change has a bootstrap 95% interval excluding zero. The intervals are very narrow because this is one fixed seed with many nearly deterministic parallel simulations; they quantify repeatability under this protocol, not generalization across seeds, speeds, or terrain.

## Conclusion

The fine-tuning **did improve the behavior it was intended to address**: foot usage became more balanced, LF–RH synchronization improved strongly, and the two diagonal groups alternated almost perfectly. However, it did **not** improve every definition of trot quality. Touchdown phases became less precise, RF–LH agreement declined, stride timing became slightly more variable, and the aggregate TDI became 2.1% worse.

Therefore, the defensible conclusion is:

> The four-foot fine-tuning corrected the asymmetric/contact-structure failure, but it traded away some temporal precision. It is a useful intermediate policy, not yet a globally better natural trot.

## Next experiment criterion

Future fine-tuning should be considered an improvement only if it preserves the observed duty-factor balance while reducing phase MAE and TDI. The next reward experiment should target phase consistency or bilateral symmetry without increasing the number of unrelated reward terms. Evaluation should then be repeated at several fixed speeds (for example 0.5, 1.0, 1.5, and 2.0 m/s) and across multiple seeds before making a general claim.

## Reproduction

Run `scripts/rsl_rl/evaluate_gait.py` once per checkpoint using identical arguments, then compare the outputs with `scripts/rsl_rl/compare_gait.py`. Raw trial values and machine-readable summaries are stored under `gait_evaluation/`.
