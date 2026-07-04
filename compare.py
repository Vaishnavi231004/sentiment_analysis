import torch
from transformers import pipeline
from datasets import load_dataset
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# -------------------------------
# Step 1: Setup device
# -------------------------------
device = 0 if torch.cuda.is_available() else -1


print(f"Device set to use: {'cuda:0' if device==0 else 'cpu'}")

# -------------------------------
# Step 2: Load Models
# -------------------------------
cardiff = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    framework="pt",
    device=device
)

bertweet = pipeline(
    "sentiment-analysis",
    model="finiteautomata/bertweet-base-sentiment-analysis",
    framework="pt",
    device=device
)

# -------------------------------
# Step 3: Load Dataset (tweet_eval sentiment subset)
# -------------------------------
dataset = load_dataset("tweet_eval", "sentiment", split="train")  # first 200 samples for testing
texts = dataset["text"]
labels_map = {0: "negative", 1: "neutral", 2: "positive"}
true_labels_str = [labels_map[l] for l in dataset["label"]]

# -------------------------------
# Step 4: Get Predictions in batches
# -------------------------------
batch_size = 64

# BERTweet - use_map=True because labels are 'pos', 'neu', 'neg'
bertweet_label_map = {
    "neg": "negative",
    "neu": "neutral",
    "pos": "positive"
}






# Updated get_predictions function
def get_predictions(model, texts, use_map=False, label_map=None):
    preds = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        outputs = model(batch_texts)
        for o in outputs:
            label = o['label']
            if use_map:
                label = label_map[label.lower()]  # convert to lowercase before mapping
            preds.append(label)
    return preds

# CardiffNLP - labels already 'positive', 'neutral', 'negative'
cardiff_preds = get_predictions(cardiff, texts, use_map=False)

# BERTweet - use mapping
bertweet_preds = get_predictions(bertweet, texts, use_map=True, label_map=bertweet_label_map)

# -------------------------------
# Step 5: Evaluate
# -------------------------------
print("=== CardiffNLP (RoBERTa) ===")
print(classification_report(true_labels_str, cardiff_preds, digits=3))

print("=== BERTweet ===")
print(classification_report(true_labels_str, bertweet_preds, digits=3))


# -------------------------------
# Step 6: Confusion Matrix
# -------------------------------
def plot_confusion_matrix(true_labels, pred_labels, model_name):
    cm = confusion_matrix(true_labels, pred_labels, labels=["negative", "neutral", "positive"])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["negative", "neutral", "positive"],
                yticklabels=["negative", "neutral", "positive"])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

# Plot for CardiffNLP
plot_confusion_matrix(true_labels_str, cardiff_preds, "CardiffNLP (RoBERTa)")

# Plot for BERTweet
plot_confusion_matrix(true_labels_str, bertweet_preds, "BERTweet")
