 # SMS Spam Detection using Machine Learning

A machine learning project that classifies SMS messages as **Spam** or **Ham** (legitimate messages) using TF-IDF feature extraction and a Multinomial Naive Bayes classifier.

## Project Overview

SMS spam can contain fraudulent offers, unwanted advertisements, and misleading claims. This project builds a text classification pipeline to distinguish spam messages from legitimate messages.

The project covers data exploration, preprocessing, feature engineering, model training, evaluation, error analysis, and custom message prediction.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Matplotlib and Seaborn
- Jupyter Notebook

## Project Workflow

1. **Data Exploration:** Examine the dataset, class distribution, and message lengths.
2. **Data Preprocessing:** Remove duplicate records, clean the dataset structure, encode labels, and split the data.
3. **Feature Engineering:** Convert SMS text into numerical TF-IDF features.
4. **Model Training:** Train a Multinomial Naive Bayes classifier.
5. **Model Evaluation:** Evaluate classification metrics, inspect the confusion matrix, and analyze missed spam messages.
6. **Custom Testing:** Predict labels and estimated spam probabilities for new messages.

## Model Performance

Baseline performance on the held-out test split:

| Metric | Score |
|---|---:|
| Accuracy | 95.16% |
| Spam Precision | 100.00% |
| Spam Recall | 61.83% |
| Spam F1-score | 76.42% |

The baseline model achieved high precision but missed some spam messages. These results demonstrate why accuracy alone is insufficient for evaluating an imbalanced classification problem.

## Project Structure

```text
sms-spam-detection/
├── data/
│   └── processed/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_model_testing.ipynb
├── src/
│   └── predict.py
├── models/
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository and enter the project directory:

```bash
git clone <YOUR_REPOSITORY_URL>
cd sms-spam-detection
```

Create and activate the Conda environment:

```bash
conda create -n spam-nb python=3.11
conda activate spam-nb
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the Prediction Script

The prediction script requires the trained model and fitted TF-IDF vectorizer in the `models/` directory.

Run:

```bash
python src/predict.py
```

Enter an SMS message when prompted. The script displays the predicted class and estimated spam probability.

## Dataset

The project uses the SMS Spam Collection dataset distributed through Kaggle:

https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Check the dataset's license and terms before redistributing the data.

## Limitations

- The model may miss spam messages with unfamiliar or ordinary-looking wording.
- Predicted probabilities are not guaranteed to be well calibrated.
- Custom-written test messages do not establish real-world accuracy unless their true labels are verified.
- The reported baseline metrics should not be confused with results from the exploratory threshold experiment.

## Future Improvements

- Compare Multinomial Naive Bayes with Logistic Regression and other text classifiers.
- Select classification thresholds using a validation set.
- Evaluate on an independent labeled dataset.
- Explore deployment through a lightweight interface or API.