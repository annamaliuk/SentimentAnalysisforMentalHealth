import re
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "../models"  # folder with the files from Drive
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- SVM ---
svm = joblib.load(f"{MODEL_DIR}/svm_model.joblib")

# --- CNN ---
class TextCNN(nn.Module):
    def __init__(self, vocab_size, n_classes, emb_dim, num_filters, kernels, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(0.2)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, num_filters, k) for k in kernels])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernels), n_classes)

    def forward(self, x):
        x = self.emb_dropout(self.emb(x)).transpose(1, 2)
        x = torch.cat([F.relu(conv(x)).max(dim=2).values for conv in self.convs], dim=1)
        return self.fc(self.dropout(x))

def tokenize(text):
    return re.findall(r"[a-z0-9']+|[^\sa-z0-9']", text.lower())

ckpt = torch.load(f"{MODEL_DIR}/cnn_model.pt", map_location=device, weights_only=False)
vocab, cnn_classes, MAX_LEN, p = ckpt["vocab"], ckpt["classes"], ckpt["max_len"], ckpt["params"]
cnn = TextCNN(len(vocab), len(cnn_classes), p["emb_dim"], p["num_filters"], p["kernels"], p["dropout"])
cnn.load_state_dict(ckpt["state_dict"])
cnn.to(device).eval()

# --- MentalRoBERTa ---
rob_dir = f"{MODEL_DIR}/mentalroberta_model"
rob_tokenizer = AutoTokenizer.from_pretrained(rob_dir)
rob = AutoModelForSequenceClassification.from_pretrained(rob_dir).to(device).eval()
rob_classes = [rob.config.id2label[i] for i in range(rob.config.num_labels)]


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def predict_svm(text):
    scores = svm.decision_function([text])[0]
    return dict(zip(svm.classes_, softmax(scores).tolist()))

@torch.no_grad()
def predict_cnn(text):
    ids = [vocab.get(w, 1) for w in tokenize(text)][:MAX_LEN]
    ids += [0] * (MAX_LEN - len(ids))
    logits = cnn(torch.tensor([ids], device=device))[0]
    return dict(zip(cnn_classes, torch.softmax(logits, 0).tolist()))

@torch.no_grad()
def predict_roberta(text):
    ids = rob_tokenizer(text)["input_ids"]
    if len(ids) > 512:
        ids = ids[:128] + ids[-384:]
    x = torch.tensor([ids], device=device)
    logits = rob(input_ids=x, attention_mask=torch.ones_like(x)).logits[0]
    return dict(zip(rob_classes, torch.softmax(logits.float(), 0).tolist()))

def classify(text):
    if not text or not text.strip():
        return None, None, None, "Enter some text first."
    results = [predict_svm(text), predict_cnn(text), predict_roberta(text)]
    top = [max(r, key=r.get) for r in results]
    if len(set(top)) == 1:
        summary = f"**All three models agree: {top[0]}**"
    else:
        summary = f"**Models disagree:** SVM predicts {top[0]}; CNN predicts {top[1]}; MentalRoBERTa predicts {top[2]}"
    return *results, summary

#
# examples = [
#     ["Had a great weekend hiking with friends, the weather was perfect and we laughed the whole time."],
#     ["I can't stop worrying about tomorrow, my heart is racing and I feel like I can't breathe properly."],
#     ["I haven't left my bed in days. Nothing feels worth doing anymore and I don't even know why I try."],
#     ["My workload is insane, my boss keeps adding deadlines and I'm completely exhausted."],
#     ["Stopped my meds last week and I've barely slept for three nights, starting five new projects at once."],
# ]
#
# with gr.Blocks(title="Mental Health Text Classifier") as demo:
#     gr.Markdown(
#         "# Mental health text classification\n"
#         "Enter atext in English from social media post. Three models trained on the same data predict one of 7 classes: Anxiety, Bipolar, Depression, Normal, Personality disorder, Stress, Suicidal. "
#     )
#     text = gr.Textbox(lines=6, label="Text", placeholder="Type or paste text here...")
#     btn = gr.Button("Classify", variant="primary")
#     summary = gr.Markdown()
#     with gr.Row():
#         out_svm = gr.Label(num_top_classes=3, label="Linear SVM + TF-IDF (test macro-F1 0.740)")
#         out_cnn = gr.Label(num_top_classes=3, label="CNN (test macro-F1 0.739)")
#         out_rob = gr.Label(num_top_classes=3, label="MentalRoBERTa (test macro-F1 0.828)")
#     gr.Examples(examples, inputs=text)
#     btn.click(classify, inputs=text, outputs=[out_svm, out_cnn, out_rob, summary])
#
# if __name__ == "__main__":
#     demo.launch()

examples = [
    ["Had a great weekend hiking with friends, the weather was perfect and we laughed the whole time."],
    ["I haven't left my bed in days. Nothing feels worth doing anymore and I don't even know why I try."],
    ["I don't see the point in living anymore. Everyone would be better off without me and I'm tired of pretending I'm okay."],
    ["I can't stop worrying about tomorrow, my heart is racing and I feel like I can't breathe properly."],
    ["Stopped my meds last week and I've barely slept for three nights, starting five new projects at once."],
    ["My workload is insane, my boss keeps adding deadlines and I'm completely exhausted."],
    ["One day I adore someone and the next I can't stand them. I never know who I really am, my therapist mentioned BPD."],
]
example_labels = ["Normal", "Depression", "Suicidal", "Anxiety", "Bipolar", "Stress", "Personality disorder"]
theme = gr.themes.Base(
    primary_hue="pink",
    neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

css = """
footer {display: none !important;}
[data-testid="block-label"], [data-testid="block-info"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding-left: 0 !important;
}
"""

theme = gr.themes.Base(
    primary_hue="pink",
    neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="*neutral_950",
    body_text_color="*neutral_100",
    body_text_color_subdued="*neutral_400",
    background_fill_primary="*neutral_900",
    background_fill_secondary="*neutral_900",
    block_background_fill="*neutral_900",
    block_border_color="*neutral_800",
    block_label_text_color="*neutral_200",
    block_title_text_color="*neutral_200",
    border_color_primary="*neutral_800",
    input_background_fill="*neutral_800",
    input_border_color="*neutral_700",
    button_secondary_background_fill="*neutral_800",
    button_secondary_text_color="*neutral_100",
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_400",
    button_primary_text_color="white",
)


with gr.Blocks(title="Mental Health Text Classifier") as demo:
    gr.Markdown(
        "# Mental health text classification\n"
        "Enter an English social media post. Three models trained on the same data predict one of 7 classes: "
        "Anxiety, Bipolar, Depression, Normal, Personality disorder, Stress, Suicidal."
    )
    text = gr.Textbox(lines=6, label="Text", placeholder="Type or paste text here...")
    btn = gr.Button("Classify", variant="primary")
    summary = gr.Markdown()
    with gr.Row():
        out_svm = gr.Label(num_top_classes=3, label="Linear SVM + TF-IDF (test macro-F1 0.740)")
        out_cnn = gr.Label(num_top_classes=3, label="CNN (test macro-F1 0.739)")
        out_rob = gr.Label(num_top_classes=3, label="MentalRoBERTa (test macro-F1 0.828)")
    gr.Examples(examples, inputs=text, example_labels=example_labels, label="Example per class")
    btn.click(classify, inputs=text, outputs=[out_svm, out_cnn, out_rob, summary])

if __name__ == "__main__":
    demo.launch(theme=theme, css=css)