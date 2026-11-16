# Mental Health Text Classification: SVM vs CNN vs MentalRoBERTa

Comparison of three NLP approaches (classical ML, a neural network and a fine-tuned transformer) on classifying social media posts into 7 mental health categories.

## Dataset

[Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) (Kaggle): English posts from Reddit and Twitter labelled with one of 7 classes: Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, Personality disorder.

- 51,022 posts after cleaning: missing values, conflicting labels and duplicates removed
- Strong class imbalance: Normal 31.4% vs Personality disorder 1.8% (17.9 : 1)
- Stratified split 70 / 15 / 15: train 35,715, validation 7,653, test 7,654

The dataset is not included in the repository. Download it from Kaggle.

## Models

| Model | Input | Key settings |
|---|---|---|
| Linear SVM | TF-IDF, words + bigrams | C = 0.3, sublinear TF, stopwords kept |
| CNN | word embeddings learned from scratch | kernels 2–5 × 300 filters, dropout 0.3, AdamW |
| MentalRoBERTa | BPE tokens, 512 max (head + tail truncation) | full fine-tuning, lr 2e-5, 3 epochs, fp16 |

Same protocol for all three: hyperparameters chosen on validation macro-F1, test set used once. Class imbalance is handled with class-weighted losses, and macro-F1 is the main metric.

## Results (test set)

| Model | Accuracy | Macro-F1 | Macro-AUC | Train time |
|---|---|---|---|---|
| Linear SVM | 0.785 | 0.740 | 0.950 | 39 s (CPU) |
| CNN | 0.787 | 0.739 | 0.960 | 578 s (T4) |
| MentalRoBERTa | 0.847 | 0.828 | 0.980 | 1,561 s (T4) |

- MentalRoBERTa outperforms both other models in every class (+8.8 macro-F1 over SVM). The difference is statistically significant (bootstrap 95% CI, McNemar p < 1e-43).
- SVM and CNN are statistically equivalent (McNemar p = 0.51).
- The hardest pair for all models is Depression vs Suicidal: even the best model labels 22.9% of Depression posts as Suicidal. This is largely due to labels coming from subreddits, not clinical assessment.

## Repository structure

```
app/
  app.py                        Gradio demo: all three models on any input text
models/                         trained models (not included, see "How to run")
notebooks/
  EDA.ipynb                     data cleaning, statistics, train/val/test split
  SVM_model.ipynb               TF-IDF + Linear SVM with grid search
  CNN_model.ipynb               Kim-style CNN in PyTorch
  MentalRoBERTa_model.ipynb     transformer fine-tuning
  results_analysis.ipynb        comparison, significance tests, ROC/AUC, error analysis
results/                        test metrics and predictions of each model (CSV)
requirements.txt
```

## How to run

**Training (Google Colab, T4 GPU recommended)**
1. Upload the dataset to Google Drive and set the path at the top of each notebook.
2. Run `EDA.ipynb` first: it creates the train/val/test splits used by all models.
3. Run `SVM_model.ipynb`, `CNN_model.ipynb` and `MentalRoBERTa_model.ipynb` (any order), then `results_analysis.ipynb`.
4. MentalRoBERTa is gated on Hugging Face: accept its terms at [mental/mental-roberta-base](https://huggingface.co/mental/mental-roberta-base) and log in with a token.
5. Each training notebook saves its model. Copy the saved models into `models/`.

**Demo (local)**
```bash
pip install -r requirements.txt
python app/app.py
```
Then open http://127.0.0.1:7860 in your browser. The `models/` folder must contain `svm_model.joblib`, `cnn_model.pt` and `mentalroberta_model/`. Model files are not included in the repository because of their size.

## Limitations

- MentalRoBERTa was pretrained on mental health subreddits, which may overlap with the dataset's sources and slightly inflate its score.
- Labels reflect where a post was published, not a diagnosis.
- This is a research project, **not a diagnostic tool**.
