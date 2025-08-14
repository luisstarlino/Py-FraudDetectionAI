# 🧠 Fraud Detecting AI - Python

![Banner de IA e Finanças](https://i.imgur.com/your-project-banner-image.png)

This project is a practical study conducted during my postgraduate studies in Machine Learning for the **detection of fraud in financial transactions**.
It simulates the development cycle of an AI model, from data generation to the training of an artificial neural network.

---

## 🚀 Project Resources

-   **Complete Pipeline:** The project demonstrates a complete workflow in Machine Learning, from data preprocessing to the evaluation of a Deep Learning model.
-   **Realistic Synthetic Dataset:** A dataset of 5,000 transactions was artificially generated with 10% fraud cases, simulating a real environment.
-   **Detailed Preprocessing:** Tune columns, categories, and location were transformed into numerical features, prepared for neural network training.
-   **Commented Code:** All steps in Jupyter Notebook are detailed, facilitating understanding of the process and replication of the project.

---

## 📊 Mehodology

The project was structured in the following stages, with their respective notebooks:

1.  **Generation of Synthetic Dataset (`01_dummy_data_generator.ipynb`)**
    -   Using the `Faker` library, we generate a dataset with realistic attributes such as transaction value, card details, payment history, etc

2.  **Exploratory Analysis and Preprocessing (`02_preprocessing_data.ipynb`)**
    -   In this phase, the raw data was transformed. This included extracting numerical features from date/time columns (e.g, time differenc in days), converting categories (`good`, `late`, `delinquent`) to numbers, and separating location coordinates.

3.  **Deep Learning Model Training (`03_training_dataset.ipynb`)**
    -   A simple artificial neural network was built using **TensorFlow/Keras**. The architecture consists of dense layers and an output layer with `sigmoid` activation for binary classification. The training use `binary_crossentropy` as the loss function and the `EarlyStopping` technique to avoid overfitting.

---

## ✅ Model Results and Evaluation

After training, the model was evaluated on a test dataset, yielding the following results:

### Accuracy
-   **Test Data Accuracy:** **67%**

### Confusion Matrix
The confusion matrix provides a more granular view of performance, highlighting the model's ability to identify each type of transaction:

![Matriz de Confusão do Modelo](https://i.imgur.com/your-confusion-matrix-image.png)

Matrix Analysis:
-   **False Negatives (FN):** [Enter value] - These represent fraudulent transactions that the model incorrectly classified as legitimate. The high rate of FN is a critical point to be improved.
-   **False Positives (FP):** [Enter value] - Legitimate transactions that the model mistakenly identified as fraud.

---

## 💡 Next steps and improvements

The 67% accuracy indicates that there is great potention for optmization. The next steps in the project will focus on improving the model:

1.  **Feature Engineering:** Create more predictive features, such as the geographic distance between consecutive transactions and the ratio of the transaction value to the user's credit limit.
2.  **Model Optimization:** Explore other more robust odels, such as **Random Forest** or **XGBoost**, which are often effective for tabular data, or adjust the hyperparameters of the curent neural network.
3.  **Class Balancing:** Implement techniques such as **SMOTE** to address the imbalance between fraud and non-fraud classes, which can significantly improve the model's ability to detect fraud.
4.  **Fraud-Focused Metrics:** Evaluate performance with more appropriate metric, such as **Recall**, **Precision** and **F1-Score**, which plac greater emphasis on accurate fraud detection.

---

## 🛠 Technologies

-   **Python 3.10+**
-   **Jupyter Notebook** (by Anaconda)
-   **Pandas**, **NumPy**
-   **Faker** (for synthetic data generation)
-   **Matplotlib**, **Seaborn** (for data analysis)
-   **TensorFlow/Keras** (for model training)
-   **scikit-learn** (for preprocessing and metrics)

---

## 📁 Project Folder and Paths

```bash
/projeto_fraude_ia
├── notebooks/
│   ├── 01_dummy_data_generator.ipynb
│   ├── 02_preprocessing_data.ipynb
│   ├── 03_training_dataset.ipynb
│   └── 04_model_using_example.ipynb
├── data/
│   ├── transacoes_fraude.csv
│   └── transacoes_fraude_preprocessado.csv
├── ai-model.keras              # My AI MODEL
├── README.md                   # This file =)
└── requirements.txt            # Project dependencies