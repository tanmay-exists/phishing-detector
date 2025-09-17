# train.py
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib


def clean_dataframe(df):
    # If "text" column not present, try to build it
    if "text" not in df.columns:
        if "subject" in df.columns and "body" in df.columns:
            df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
        elif "body" in df.columns:
            df["text"] = df["body"].fillna("")
        else:
            raise ValueError(
                "Dataset must contain either a 'text' column or ('subject' + 'body')."
            )

    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    return df[["text", "label"]]


def train_tfidf(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    X = df["text"].astype(str)
    y = df["label"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)

    print("✅ TF-IDF model val acc:", accuracy_score(y_val, preds))
    print(classification_report(y_val, preds))
    joblib.dump(pipe, os.path.join(out_dir, "tfidf_lr.joblib"))
    print("💾 Saved TF-IDF model.")


def train_transformer(df, out_dir, model_name="distilbert-base-uncased", epochs=3):
    # import only if needed
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    import torch
    from datasets import Dataset
    import evaluate

    os.makedirs(out_dir, exist_ok=True)
    ds = Dataset.from_pandas(df[["text", "label"]])
    ds = ds.train_test_split(test_size=0.1, stratify_by_column="label", seed=42)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    ds = ds.map(tokenize, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=os.path.join(out_dir, "hf_out"),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return metric.compute(predictions=preds, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        compute_metrics=compute_metrics
    )
    trainer.train()
    model.save_pretrained(os.path.join(out_dir, "transformer_model"))
    tokenizer.save_pretrained(os.path.join(out_dir, "transformer_model"))
    print("💾 Saved transformer model and tokenizer.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/emails.csv")
    parser.add_argument("--out_dir", default="models")
    parser.add_argument("--method", choices=["tfidf", "transformer", "both"], default="tfidf")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = clean_dataframe(df)

    if args.method in ("tfidf", "both"):
        train_tfidf(df, args.out_dir)
    if args.method in ("transformer", "both"):
        train_transformer(df, args.out_dir, epochs=args.epochs)
