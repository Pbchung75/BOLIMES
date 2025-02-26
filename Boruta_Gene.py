import os
import logging
import warnings
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

# Ignore warnings
warnings.filterwarnings('ignore')

datasets = [20685, 20711, 21050, 21122, 29354, 30784, 31312, 31552, 32537, 33315, 36895, 37364, 39582]
#folders = '01.07.BestF'
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, header=None, sep='\s+')
        logging.info(f"Reading data from file: {file_path}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
def scale_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info("Data has been standardized.")
    return X_scaled

# Function to select features using Boruta
def select_features_with_boruta(X_scaled, y):
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    boruta_selector = BorutaPy(
        rf,
        n_estimators=300,
        max_iter=200,
        alpha=0.01,
        perc=100,
        two_step=True,
        random_state=42,
        verbose=1
    )
    start_time = time.time()
    boruta_selector.fit(X_scaled, y)
    end_time = time.time()

    feature_selection_time = end_time - start_time
    logging.info(f"Boruta feature selection time: {feature_selection_time:.2f} seconds")
    print(f"Boruta feature selection time: {feature_selection_time:.2f} seconds")

    X_selected = X_scaled[:, boruta_selector.support_]
    num_selected_features = boruta_selector.support_.sum()
    logging.info(f"Number of selected features: {num_selected_features}")
    return X_selected, boruta_selector, num_selected_features, feature_selection_time

# Save the selected feature names to a CSV file
def save_selected_feature_names_to_csv(X, selected_features, output_directory, filename='selected_feature_names.csv'):
    selected_feature_names = X.columns[selected_features]
    selected_features_file = os.path.join(output_directory, filename)
    pd.DataFrame(selected_feature_names, columns=['Selected Features']).to_csv(selected_features_file, index=False)
    logging.info(f"Selected feature names saved to file: {selected_features_file}")
    print(f"Selected feature names saved to file: {selected_features_file}")

# Save Boruta summary to a CSV file
def save_boruta_summary_to_csv(boruta_selector, feature_selection_time, output_directory, filename='boruta_summary.csv'):
    # Count Confirmed, Tentative, and Rejected features
    confirmed_count = boruta_selector.support_.sum()
    tentative_count = boruta_selector.support_weak_.sum()
    rejected_count = (~(boruta_selector.support_ | boruta_selector.support_weak_)).sum()
    summary_df = pd.DataFrame({
        'Confirmed': [confirmed_count],
        'Tentative': [tentative_count],
        'Rejected': [rejected_count],
        'Feature Selection Time (s)': [feature_selection_time]
    })
    summary_file = os.path.join(output_directory, filename)
    summary_df.to_csv(summary_file, index=False)
    logging.info(f"Boruta summary saved to file: {summary_file}")
    print(f"Boruta summary saved to file: {summary_file}")
all_results = pd.DataFrame(columns=['Dataset', 'Confirmed', 'Tentative', 'Rejected', 'Feature Selection Time (s)'])
for dataset in datasets:
    file_path = f'Gene Data/{dataset}/data.trn.gz'

    output_directory = os.path.join('Experiment', 'Selected Features', 'Results', folders, str(dataset))

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    log_file = os.path.join(output_directory, 'experiment_log.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Starting feature selection process with Boruta for dataset: {dataset}.")
    df = load_data(file_path)
    if df is not None:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_scaled = scale_data(X)

        X_selected, boruta_selector, num_selected_features, feature_selection_time = select_features_with_boruta(X_scaled, y)
        print(f"Number of selected features for dataset {dataset}: {num_selected_features}")

        save_selected_feature_names_to_csv(X, boruta_selector.support_, output_directory)
        save_boruta_summary_to_csv(boruta_selector, feature_selection_time, output_directory)

        confirmed_count = boruta_selector.support_.sum()
        tentative_count = boruta_selector.support_weak_.sum()
        rejected_count = (~(boruta_selector.support_ | boruta_selector.support_weak_)).sum()

        all_results = pd.concat([all_results, pd.DataFrame([{
            'Dataset': dataset,
            'Confirmed': confirmed_count,
            'Tentative': tentative_count,
            'Rejected': rejected_count,
            'Feature Selection Time (s)': feature_selection_time
        }])], ignore_index=True)

output_all_file = os.path.join('Experiment', 'Selected Features', 'Results', folders, 'ALL_boruta_summary.csv')
all_results.to_csv(output_all_file, index=False)
logging.info(f"Combined Boruta results for all datasets saved to file: {output_all_file}")
print(f"Combined Boruta results for all datasets saved to file: {output_all_file}")
