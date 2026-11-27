# JEV Classifier Benchmark: December 1–14

## Week 1: Implementation and Baselines

### December 1–3: Setup and Traditional ML (IN PROGRESS)
- [ ] `01_dataset_exploration.ipynb`
  - Prepare all four datasets.
  - Verify train/test splits, labels, and class distributions.
- [ ] `02_baseline_experiments.ipynb`
  - Implement TF-IDF + Logistic Regression.
  - Implement TF-IDF + LinearSVC.
  - Implement TF-IDF + Naive Bayes.
  - Run traditional ML baselines across all datasets.

### December 4–5: JEV and DistilBERT
- [ ] `03_jev_experiments.ipynb`
  - Integrate JEV API.
  - Test classification and confidence outputs.
  - Run JEV across all four datasets.
  - Collect accuracy, Macro-F1, and API costs.
- [ ] `04_transformer_experiments.ipynb`
  - Fine-tune DistilBERT across all four datasets.
  - Collect predictions and metrics.

### December 6–7: SetFit and Initial Results
- [ ] `04_transformer_experiments.ipynb`
  - Implement SetFit.
  - Run experiments with 8, 16, 32, and 100 examples per class.
- [ ] Review initial results.
  - Consolidate model results.
  - Verify experimental consistency.
  - Identify missing experiments.

## Week 2: Evaluation and Final Report

### December 8–9: Calibration and Performance
- [ ] `06_calibration_analysis.ipynb`
  - Calculate ECE and Brier Score.
  - Generate calibration diagrams.
  - Compare model confidence.
- [ ] `07_inference_benchmarks.ipynb`
  - Benchmark inference latency.
  - Measure P95 latency and throughput.

### December 10–11: Cost and Training Data Analysis
- [ ] `07_inference_benchmarks.ipynb`
  - Calculate cost per 1,000 predictions.
  - Compare inference costs across models.
- [ ] `05_low_data_experiments.ipynb`
  - Analyze training data size experiments.
  - Compare performance at 8, 16, 32, and 100 examples per class.
  - Generate performance graphs.

### December 12–13: Abstention and Report Draft
- [ ] `08_abstention_analysis.ipynb`
  - Implement confidence-based abstention.
  - Evaluate coverage versus accuracy.
  - Investigate achieving 90–95% accuracy.
- [ ] `09_final_comparison.ipynb`
  - Consolidate results and comparison tables.
  - Generate final visualizations.
- [ ] Draft final report.

### December 14: Final Review
- [ ] Finalize report.
- [ ] Review methodology and results.
- [ ] Verify citations and references (minimum 5).
- [ ] Ensure notebooks and experiments are accessible and reproducible.

---

## Models
- JEV
- TF-IDF + Logistic Regression
- TF-IDF + LinearSVC
- TF-IDF + Naive Bayes
- DistilBERT
- SetFit
- LLM classifier (optional)

## Datasets
- 20 Newsgroups
- AG News
- Banking77
- IMDb Reviews (proposed fourth dataset)

## Evaluation Metrics
- Accuracy
- Macro-F1
- Expected Calibration Error (ECE)
- Brier Score
- P95 Latency
- Throughput
- Cost per 1,000 predictions

## Additional Experiments
- Training data size: 8, 16, 32, and 100 examples per class.
- Confidence-based abstention.
- Coverage versus selective accuracy.

## Final Report

- 10–15 pages.
- Double-spaced.
- Times New Roman, 12 pt.
- Standard journal format.
- Minimum 5 references.

### Required Sections
1. Abstract
2. Introduction
3. Literature Review and Critique
4. Methodology and Experimental Setup
5. Results
6. Discussion and Conclusions
7. References

## Code and Data Deliverables
- Provide access to all notebooks.
- Include datasets or instructions for accessing them.
- Include code, configurations, and experimental results.
- Ensure experiments are reproducible.