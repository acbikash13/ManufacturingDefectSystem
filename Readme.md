

# Automated Coil Defect Detection System

This repository contains the implementation of an **Automated Coil Defect Detection System**, 
designed to analyze and classify defects in coil materials for industrial purposes.

## Overview

This project leverages machine learning algorithms and deep learning techniques to identify defect patterns 
in coil production. By analyzing data from materials, the system aids in quality control and reduces production losses.

## Features
- Preprocessing and feature engineering from industrial defect datasets.
- Implementation of multiple machine learning models for classification and anomaly detection:
  - Isolation Forest (IForest)
  - K-Nearest Neighbors (KNN)
  - Decision Tree Classifier
  - Support Vector Classifier (SVC)
  - Neural Networks using TensorFlow and Keras
- Model evaluation and hyperparameter optimization using cross-validation and grid search.
- Visualization of decision trees and performance metrics.
- Real-time defect detection and classification pipeline.

## Techniques and Algorithms

### Preprocessing
- **Tools used**: Pandas, NumPy, Scikit-learn.
- Steps:
  - Label encoding for categorical data.
  - Standardization and normalization of features.
  - Splitting data into training and test sets using `train_test_split`.

### Machine Learning Models
1. **Isolation Forest (IForest)**:
   - Anomaly detection for identifying outliers in the dataset.
2. **K-Nearest Neighbors (KNN)**:
   - Classification based on the nearest neighbors in feature space.
3. **Decision Tree Classifier**:
   - A visual and interpretable approach to defect classification.
4. **Support Vector Classifier (SVC)**:
   - A powerful algorithm for classification tasks with a high-dimensional space.
5. **Neural Networks (Keras & TensorFlow)**:
   - Custom-built Sequential models for binary and multi-class classification.

### Evaluation Metrics
- Accuracy
- F1 Score
- Mean Squared Error
- R² Score
- Classification Reports

## Results
- The neural network achieved **90%+ accuracy** after rigorous cross-validation and hyperparameter tuning.
- Models like Decision Trees provided interpretable results, while anomaly detection techniques highlighted key defect patterns.
- The overall system successfully identified defects, with significant potential for industrial deployment.

## Requirements

Install the necessary dependencies using:

```bash
pip install -r requirements.txt
```

### Key Libraries:
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- PyOD
- Matplotlib

## Project Structure
```
.
├── Automated_Coil_Detection_System.ipynb   # Main Jupyter notebook
├── export_data_AP4_Defect.csv             # Dataset
├── requirements.txt                       # List of dependencies
└── README.md                              # Project documentation
```

## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/automated-coil-detection-system.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook to train models and visualize results:
   ```bash
   jupyter notebook Automated_Coil_Detection_System.ipynb
   ```

## Future Work
- Integration of real-time streaming data.
- Deployment as a web service using Flask or FastAPI.
- Incorporation of additional data sources for enhanced model training.

## Contributing
Contributions, issues, and feature requests are welcome! Feel free to fork the repository and create a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

