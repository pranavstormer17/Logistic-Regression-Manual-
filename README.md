# Diabetes Prediction Model

This script trains a Logistic Regression machine learning model to predict diabetes outcomes based on diagnostic measurements.

## Dataset

The code uses the Pima Indians Diabetes Database. You can download the CSV file from Plotly's datasets repository here:

[plotly/datasets/master/diabetes.csv](https://github.com/plotly/datasets/blob/master/diabetes.csv)

## Dependencies

To run this script, you will need the following Python libraries installed:

* `pandas`
* `scikit-learn`
* `seaborn`
* `matplotlib`

You can install them via pip:

```bash
pip install pandas scikit-learn seaborn matplotlib

```

## Usage

1. Download the `diabetes.csv` file from the link above.
2. **Important:** Open the script and update the file path to match where you saved the dataset on your local machine (replacing `/home/pranavstormer17/Downloads/diabetes.csv`):
```python
data = pd.read_csv('/path/to/your/diabetes.csv', header=0, names=col_names)

```


3. Run the script.

## Output

When executed, the script will:

* Print basic dataset statistics (shape, first 5 rows, null value check).
* Split the data into 80% training and 20% testing sets.
* Output the model's Confusion Matrix and Accuracy Score to the console.
* Generate and display an interactive Seaborn heatmap of the Confusion Matrix.
