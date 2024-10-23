import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, BertTokenizer
from tqdm import tqdm
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification,TFBertForSequenceClassification, TFTrainingArguments
import os


data_df = pd.read_csv("20_newsgroup.csv")
data_df = data_df.drop_duplicates(subset="text").drop(columns="Unnamed: 0")
data_df['text'] = data_df['text'].astype(str)
data_df = data_df.drop(data_df[data_df['text'].str.strip() == "nan"].index)
print(data_df)

data_texts = data_df['text'].to_list()
data_labels = data_df['target'].to_list()
train_texts, val_texts, train_labels, val_labels = train_test_split(data_texts, data_labels, test_size = 0.2, random_state = 0 )
print(len(set(train_texts).intersection(set(val_texts))))
train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts, train_labels, test_size = 0.01, random_state = 0 )
print(len(set(train_texts).intersection(set(test_texts))))

def validate_and_tokenize(texts):
    valid_texts = []
    for i, text in enumerate(texts):
        if isinstance(text, str):
            valid_texts.append(text)
        else:
            print(f"Invalid input dataset at index {i}: {text}")
    
    if not valid_texts:
        raise ValueError(f"No valid texts found in the dataset")
    
print(validate_and_tokenize(train_texts))
print(validate_and_tokenize(val_texts))

def tokenize(model_name,model_tokenizer,batch_size=16):
    tokenizer = model_tokenizer.from_pretrained(model_name)
    train_encodings = tokenizer(train_texts, truncation = True, padding = True  )
    val_encodings = tokenizer(val_texts, truncation = True, padding = True )

    train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
    ))

    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(val_encodings),
        val_labels
    ))

    train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Print dataset info
    print("Train dataset:", train_dataset)
    print("Train dataset size:", tf.data.experimental.cardinality(train_dataset).numpy())
    print("Validation dataset:", val_dataset)
    print("Validation dataset size:", tf.data.experimental.cardinality(val_dataset).numpy())

    # Check for empty datasets
    if tf.data.experimental.cardinality(train_dataset).numpy() == 0:
        raise ValueError("Train dataset is empty")
    if tf.data.experimental.cardinality(val_dataset).numpy() == 0:
        raise ValueError("Validation dataset is empty")

    return train_encodings, val_encodings, train_dataset, val_dataset

train_encoding_distil, val_encoding_distil, train_data_distil, val_data_distil = tokenize("distilbert-base-uncased",DistilBertTokenizer)
train_encoding_bert, val_encoding_bert, train_data_bert, val_data_bert = tokenize("bert-base-uncased",BertTokenizer)

distilbert_model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=20)
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=20)

def training(modelForSequenceClassification, model_base_uncased, train_dataset, val_dataset):
    # Create the model
    trainer_model = modelForSequenceClassification.from_pretrained(model_base_uncased, num_labels=20)

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    
    trainer_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs')
    ]

    # Train the model
    history = trainer_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=7,
        callbacks=callbacks
    )

    # Evaluate the model
    eval_results = trainer_model.evaluate(val_dataset)
    print(f"Evaluation results: {eval_results}")

    # Save the model
    save_directory = "./saved_models"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    trainer_model.save_pretrained(save_directory)

    return trainer_model, history

# Usage
trained_model_distil, training_history_distil = training(TFDistilBertForSequenceClassification, 'distilbert-base-uncased', train_data_distil, val_data_distil)
trained_model_bert, training_history_bert = training(TFBertForSequenceClassification, 'bert-base-uncased', train_data_bert, val_data_bert)
