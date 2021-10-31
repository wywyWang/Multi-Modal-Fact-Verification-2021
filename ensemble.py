import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

model_path = {
    'deberta_base': 'sequence_model/model/20211028-110027_/answer.csv',
    'xlm_roberta_base': 'my_model/model/20211029-215340_/answer.csv'
}
model_count = len(model_path.keys())

inverse_category = {
    0: 'Support_Multimodal',
    1: 'Support_Text',
    2: 'Insufficient_Multimodal',
    3: 'Insufficient_Text',
    4: 'Refute'
}

# read all predictions
df = {}
for key, value in model_path.items():
    df[key] = pd.read_csv(model_path[key])

# ensemble
answer = []
for i in tqdm(range(len(df['deberta_base']))):
    prob = []
    # 0 is index in dataframe
    for prob_1, prob_2 in zip(df['deberta_base'].iloc[i].values.tolist()[1:], df['xlm_roberta_base'].iloc[i].values.tolist()[1:]):
        prob.append((prob_1+prob_2)/model_count)

    category = prob.index(max(prob))
    answer.append(inverse_category[category])

assert len(answer) == len(df['deberta_base'])
answer = pd.DataFrame(answer, columns=['Category'])
answer.to_csv('answer.csv')


df_val = pd.read_csv('./data/val.csv', index_col='Id')
df_predicted = pd.read_csv('answer.csv')

df_val['predict'] = df_predicted['Category'].values

labels = ['Support_Multimodal', 'Support_Text', 'Insufficient_Multimodal', 'Insufficient_Text', 'Refute']
a = confusion_matrix(df_val['Category'], df_val['predict'], labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=a, display_labels=labels)
print(f1_score(df_val['Category'], df_predicted['Category'], average='weighted'))