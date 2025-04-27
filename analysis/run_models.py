from sklearn.model_selection import train_test_split
import numpy as np
from typing import Callable, Dict, List
from utils import get_features
import json

def evaluate_models_on_datasets(
    dir:str,
    classifiers: Dict[str, Callable],
    num_datasets: int = 20,
    num_seeds: int = 5,
    test_size: float = 0.3
) -> Dict[str, Dict[str, List[float]]]:
    """
    Evaluate multiple models on multiple datasets using the specified feature extractor.

    Args:
        dir (str): The root directory containing the image files organized in subfolders.
        classifiers: Dictionary mapping model names to classifier constructors
        num_datasets: number of datasets to loop over
        num_seeds: Number of random splits for train/test evaluation
        test_size: Fraction of the dataset to reserve for testing

    Returns:
        results: Dictionary with structure:
                 { model_name: {'train': [...], 'test': [...]} }
    """
    results = {name: {'train': [], 'test': []} for name in classifiers}

    for j in range(20, 20+num_datasets):
        print(f"Evaluating dataset {j}...")
        features, labels, _ = get_features(dir, seed=j, num_subjects=3)

        for seed in range(num_seeds):
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=test_size, random_state=seed
            )

            for name, clf_fn in classifiers.items():
                clf = clf_fn()
                clf.fit(X_train, y_train)
                train_score = clf.score(X_train, y_train)
                test_score = clf.score(X_test, y_test)
                results[name]['train'].append(train_score)
                results[name]['test'].append(test_score)

    return results

def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    return obj

def save_results_to_json(results, filename="/Users/daixinming/Projects/AppliedStatisticsII/applied_stats_project/analysis/results.json"):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4, default=convert_to_json_serializable)

if __name__ == "__main__":
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    classifiers = {
    'SVM': lambda: OneVsRestClassifier(SVC(kernel='rbf')),
    'LogReg': lambda: LogisticRegression(penalty='l1', solver='saga', max_iter=1000, random_state=42),
    'QDA': lambda: QuadraticDiscriminantAnalysis(reg_param=0.001)
    }

    dir = '/Users/daixinming/Projects/AppliedStatisticsII/applied_stats_project/data/sd302/images/challengers/A/roll/img_l2_feature_extractions_png/img_l1_feature_extractions/freq'
    results = evaluate_models_on_datasets(dir, classifiers=classifiers)
    save_results_to_json(results)
    # Print average scores
    for model_name, scores in results.items():
        print(f"{model_name}: Train = {np.mean(scores['train']):.4f}, Test = {np.mean(scores['test']):.4f}")
