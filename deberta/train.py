from simpletransformers.classification import ClassificationModel, ClassificationArgs
import sklearn
import pandas as pd
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


MODEL_TYPE = "deberta"
PRETRAINED_PATH = 'microsoft/deberta-base'
OUTPUT_PATH = './models/deberta_base_1/'


if __name__ == '__main__':
    # df_train = read_json('./processed_data/preprocess_train.json')[['text', 'reply', 'label']].sample(frac=1).reset_index(drop=True)
    # df_dev = read_json('./processed_data/preprocess_my_dev.json')[['text', 'reply', 'label']]

    df_train = pd.read_csv('../data/train.csv')[['claim', 'document', 'Category']]
    df_dev = pd.read_csv('../data/val.csv')[['claim', 'document', 'Category']]

    category = {
        'Support_Multimodal': 0,
        'Support_Text': 1,
        'Insufficient_Multimodal': 2,
        'Insufficient_Text': 3,
        'Refute': 4
    }

    df_train['Category'] = df_train['Category'].map(category)
    df_dev['Category'] = df_dev['Category'].map(category)
    df_train.columns = ['text_a', 'text_b', 'labels']
    df_dev.columns = ['text_a', 'text_b', 'labels']

    model_args = ClassificationArgs(
        num_train_epochs=3,
        train_batch_size=8,
        eval_batch_size=8,
        max_seq_length=113,
        evaluate_during_training=True,
        evaluate_during_training_steps=2000,
        learning_rate=5e-7,
        tensorboard_dir=OUTPUT_PATH,

        output_dir=OUTPUT_PATH,
        manual_seed=42,
        use_multiprocessing=False,
        save_steps=2000,
        n_gpu=1
    )
    model = ClassificationModel(MODEL_TYPE, PRETRAINED_PATH, num_labels=5, args=model_args)

    model.train_model(df_train, eval_df=df_dev, f1=sklearn.metrics.f1_score, f1_macro=lambda truth, predictions: sklearn.metrics.f1_score(truth, predictions, average='weighted'), f1_micro=lambda truth, predictions: sklearn.metrics.f1_score(truth, predictions, average='micro'))
