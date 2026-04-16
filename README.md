# Vision Transformer (ViT) for Cats vs. Dogs Classification

Fine-tuned a pre-trained Vision Transformer (`google/vit-base-patch16-224`) on the Cats vs. Dogs dataset for binary image classification, and deployed the trained model as a Streamlit web app.

## Results

| Metric        | Value  |
|---------------|--------|
| Test Accuracy | 99.60% |
| Macro F1      | 1.00   |
| Precision     | 1.00   |
| Recall        | 1.00   |

Evaluated on 2,500 held-out images (1,250 cats + 1,250 dogs).

**Confusion matrix:**

|              | Predicted Cat | Predicted Dog |
|--------------|---------------|---------------|
| Actual Cat   | 1247          | 3             |
| Actual Dog   | 7             | 1243          |

## Dataset

- **Cats vs. Dogs** — 24,998 valid images after filtering corrupted files
- Splits: 19,998 train / 2,500 validation / 2,500 test

## Approach

- Loaded and filtered the raw PetImages dataset, removing corrupted files
- Used the ViT image processor for resizing (224×224) and ImageNet normalization
- Fine-tuned all ViT layers for binary classification (cat vs. dog)
- Optimizer: AdamW, learning rate 2e-5
- Loss: Cross-entropy
- Trained for 3 epochs with best-validation checkpointing
- Deployed the final model as a Streamlit app for single-image inference

## Stack

PyTorch · HuggingFace Transformers · scikit-learn · Streamlit · Matplotlib

## Files

- `vit_cats_vs_dogs.ipynb` — end-to-end notebook (data loading, training, evaluation, plots)

## How to Run

1. Download the Cats vs. Dogs dataset from [Microsoft](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
2. Place `PetImages.zip` in the project root
3. Install dependencies:
```bash
