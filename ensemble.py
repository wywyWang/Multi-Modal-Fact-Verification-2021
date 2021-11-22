import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

model_path = {
    'deberta_base': 'sequence_model/model/20211101-221315_/answer.csv',
    # 'deberta_base_2': 'my_model/model/20211102-164513_/answer.csv',   # worse
    'xlm_roberta_base': 'my_model/model/20211101-220613_/answer.csv',
    'roberta_base': 'sequence_model/model/20211102-131452_/answer.csv',
    # 'dino_roberta': 'sequence_model/model/20211103-003319_/answer.csv',
    # 'dino_deberta': 'my_model/model/20211103-003303_/answer.csv',
    'deberta_base_mish': 'my_model/model/20211103-124904_/answer.csv',
    'roberta_base_mish': 'sequence_model/model/20211103-124901_/answer.csv',
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
ensemble_weight = [0.6, 0.2, 0.1, 0.2, 0.3]
POWER = 1/2
for i in tqdm(range(len(df['deberta_base']))):
    prob = []
    if model_count == 1:
        for prob_1 in zip(df['deberta_base'].iloc[i].values.tolist()[1:]):
            prob.append(prob_1)
    else:
        # 0 is index in dataframe
        for prob_1, prob_2, prob_3, prob_4, prob_5 in \
            zip(df['deberta_base'].iloc[i].values.tolist()[1:], df['roberta_base'].iloc[i].values.tolist()[1:], df['xlm_roberta_base'].iloc[i].values.tolist()[1:], df['deberta_base_mish'].iloc[i].values.tolist()[1:], df['roberta_base_mish'].iloc[i].values.tolist()[1:]):
            current_prob = (prob_1**POWER) * ensemble_weight[0] + (prob_2**POWER) * ensemble_weight[1] + (prob_3**POWER) * ensemble_weight[2] + (prob_4**POWER) * ensemble_weight[3] + (prob_5**POWER) * ensemble_weight[4]
            prob.append(current_prob)

    category = prob.index(max(prob))
    answer.append([i+1, inverse_category[category]])

assert len(answer) == len(df['deberta_base'])
answer = pd.DataFrame(answer, columns=['Id', 'Category'])
answer.to_csv('answer.csv', index=False)


# df_val = pd.read_csv('./data/val.csv', index_col='Id')
df_val = pd.read_csv('./data/test.csv', index_col='Id')
df_predicted = pd.read_csv('answer.csv')

df_val['predict'] = df_predicted['Category'].values

labels = ['Support_Multimodal', 'Support_Text', 'Insufficient_Multimodal', 'Insufficient_Text', 'Refute']
# a = confusion_matrix(df_val['Category'], df_val['predict'], labels=labels)
# disp = ConfusionMatrixDisplay(confusion_matrix=a, display_labels=labels)
# print(f1_score(df_val['Category'], df_predicted['Category'], average='weighted'))