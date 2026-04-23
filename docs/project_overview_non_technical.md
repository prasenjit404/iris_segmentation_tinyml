# Project Overview (Non-Technical)

## What this project is
This project builds a practical system that looks at an eye image and creates two compact summaries:
1. one summary for the iris
2. one summary for the white part of the eye (sclera)

These summaries are saved as small tables of numbers. They are designed so other systems can use them later for recognition, analysis, or quality checks.

## Why this project is needed
Eye-image systems are often hard to deploy in the real world because they can be heavy, expensive to label, and difficult to run on smaller devices. This project addresses those issues by focusing on:
- lightweight processing
- practical labeling workflow
- clear repeatable steps from data to deployment

In short, this is not just a model experiment. It is a complete workflow that can be run, reviewed, and improved over time.

## What we are doing
We take eye images, find important regions (iris and sclera), and convert those regions into fixed-size number tables. We also provide a simple web app where someone can upload an image and download those tables.

## How we are doing it
The project is split into clear phases:
1. Understand and clean the dataset.
2. Generate initial labels automatically and send uncertain cases for review.
3. Train a small model that can run in constrained environments.
4. Run inference to produce the two final matrices.
5. Improve data quality with an active-learning cycle.
6. Test stability under different image conditions.
7. Export deployment-friendly artifacts.

This phased approach keeps the work organized and makes progress measurable.

## Key decisions we made and why
1. We enforce a local virtual environment every time.
Reason: reproducibility and fewer setup conflicts.

2. We separate people between training and testing groups.
Reason: fair evaluation and reduced risk of misleadingly high results.

3. We use assisted labeling instead of fully manual labeling from day one.
Reason: faster progress while keeping human quality control.

4. We start with a lightweight TinyML-aligned model.
Reason: easier deployment path and lower compute requirements.

5. We keep fallback behavior for difficult images.
Reason: avoid total failures and ensure outputs are still produced.

6. We test robustness, not only best-case accuracy.
Reason: real-world images are noisy, blurry, and inconsistent.

## Why this matters beyond research
Many projects stop at a demo model. This project includes:
- repeatable scripts
- clear outputs
- user-facing interface
- deployment export files

That makes it suitable for practical handoff, operational testing, and future scaling.

## Who can understand and use this
- Technical teams can use the scripts and code.
- Non-technical stakeholders can use the web interface and read summary documents.
- Project managers can track progress phase-by-phase and see what is production-ready.

## Current outcome
Today, the system can process an uploaded eye image and produce:
- iris matrix download (CSV)
- sclera matrix download (CSV)
- segmentation preview and summary information

So the project already delivers a complete path from image input to usable outputs.
