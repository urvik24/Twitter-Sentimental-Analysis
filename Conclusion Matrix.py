import pandas as pd
from sklearn.metrics import f1_score

df = pd.read_excel(r"Tweet.xlsx")
df_confusion = pd.crosstab(df['Sentiment'], df['Self'],rownames=['Actual'], colnames=['Predicted'])
print(df_confusion)
