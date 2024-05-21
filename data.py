import pandas as pd
from sklearn.model_selection import train_test_split
import csv
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

# Load the original CSV file
df = pd.read_csv("/home/sojitra_2211mc15/Daivik/TextTime/fact-update-train-over.csv")
# df = df.iloc[1:]


# Perform random oversampling on the minority class

sm = SMOTE(
    sampling_strategy='auto',  # samples only the minority class
    random_state=0,  # for reproducibility
    k_neighbors=3,
    n_jobs=2
)
enn = EditedNearestNeighbours(
    sampling_strategy='auto',
    n_neighbors=3,
    kind_sel='all',
    n_jobs=2)


smenn = SMOTEENN(
    sampling_strategy='auto',  # samples only the minority class
    random_state=0,  # for reproducibility
    smote=sm,
    enn=enn,
    n_jobs=4
)

X_train_oversampled, y_train_oversampled=smenn.fit_resample(df["text"], df["label"])

filepathV2 = "fact-update-train-oversampled.csv"
f = csv.writer(open(filepathV2, "w+"))

f.writerow(["text", "label"])


for line in range(len(X_train_oversampled)):
    txt = X_train_oversampled[line]
    label = y_train_oversampled[line]

    f.writerow([txt, label])


