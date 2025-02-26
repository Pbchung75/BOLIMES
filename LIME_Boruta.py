# Use LIME to calculate feature importance across all datasets
import os
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

datasets = ['20685', '20711', '21050', '21122', '29354', '30784', '31312', '31552', '32537', '33315', '37364', '39582', '39716']
folder = "01.07.BestF"
# Set a reasonable number of samples for LIME
num_samples = 10000
for dataset in datasets:
    print(f"Processing dataset {dataset}...")
    # Read the selected feature file
    selected_features_file = os.path.join('Experiment', 'Selected Features', 'Results', folder, dataset, 'selected_feature_names.csv')
    if not os.path.exists(selected_features_file):
        print(f"File {selected_features_file} does not exist, skipping dataset {dataset}")
        continue
    selected_features = pd.read_csv(selected_features_file)['Selected Features'].values
    # Read data from file
    file_path = f'Gene Data/{dataset}/data.trn.gz'
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist, skipping dataset {dataset}")
        continue
    df = pd.read_csv(file_path, header=None, sep='\s+')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    # Map features
    X_selected = X[selected_features]

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)
    # Initialize LIME explainer with discretize_continuous=False
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=selected_features,
        class_names=np.unique(y_train).astype(str),
        mode='classification',
        discretize_continuous=False
    )
    num_features = len(selected_features)

    aggregate_importance_df = pd.DataFrame(columns=selected_features)
    for i in range(len(X_test)):
        # Explain each individual instance
        lime_explanation = explainer.explain_instance(
            X_test.iloc[i].values,
            rf_model.predict_proba,
            num_samples=num_samples,
            num_features=num_features
        )
        lime_importance = dict(lime_explanation.as_list())
        for feature in selected_features:
            if feature in lime_importance:
                aggregate_importance_df.at[i, feature] = lime_importance[feature]
            else:
                aggregate_importance_df.at[i, feature] = 0

    mean_importance_df = aggregate_importance_df.mean().reset_index()
    mean_importance_df.columns = ['Feature', 'Mean Importance']
    mean_importance_df['Absolute Importance'] = mean_importance_df['Mean Importance'].abs()
    mean_importance_df = mean_importance_df.sort_values(by='Absolute Importance', ascending=False)
    output_folder = os.path.join('Experiment', 'Selected Features', 'Results', folder, dataset)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    lime_cleaned_output_file = os.path.join(output_folder, 'lime_mean_feature_importance.csv')
    mean_importance_df.to_csv(lime_cleaned_output_file, index=False)

    print(f"LIME average feature importance results for dataset {dataset} have been saved to: {lime_cleaned_output_file}")
