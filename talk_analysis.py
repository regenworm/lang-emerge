import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import json

# talk_location = 'models/hardy-surf-48/tasks_chatlog-test_Tue-31-May-2022-12 50 45_100H_0.0100lr_False_4_3.json'
# talk_location = 'models/hardy-surf-48/tasks_chatlog-train_Tue-31-May-2022-12 50 45_100H_0.0100lr_False_4_3.json'
talk_location = 'content/lang-emerge/models/tasks_chatlog-train_Wed-01-Jun-2022-11:59:33_50H_0.0100lr_False_4_3.json'
talk = {}
with open(talk_location, 'r') as f:
    talk = json.load(f)

# talk_gt_pred: (batch x task_size * 2)
# talk_chats: (batch x num_rounds * 2)
talk_gt_pred = np.array([[t['gt'] + t['pred']] for t in talk]).squeeze()
print(talk_gt_pred[0], talk_gt_pred.shape)
talk_chats = np.array([t['chat'] for t in talk])
print(talk_chats[0], talk_chats.shape)

# create data frame for talks
# df_cols = task_size * 2 + num_rounds * num_agents
# talk_df: (batch x df_cols)
talk_df = np.concatenate([talk_gt_pred, talk_chats], axis=1)
print(talk_df[0], talk_df.shape)

column_names_gt_pred = [prefix + t for t in talk[0]['task'] for prefix in ['gt_', 'pred_']]
column_names_chat = [f'round_{r_num} {agent}_agent' for r_num in range(len(talk_chats[0])//2 ) for agent in ['q', 'a'] ]
columns = column_names_gt_pred + column_names_chat
print(columns)

df = pd.DataFrame(data=talk_df, columns=columns)
print(df.head())


labelencoder=LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])

plt.figure(figsize=(14,12))
sns.heatmap(df.corr(method='spearman'),linewidths=.1,cmap="YlGnBu", annot=True)
plt.yticks(rotation=0);
plt.show()


