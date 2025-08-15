# 📊 Data Refiner App

An interactive **data cleaning, profiling, and prediction platform** built with **Streamlit**.  
We can quickly upload, explore, clean, and run machine learning predictions from a browser-based interface.

---

## 🚀 Features

The app is organized into **multiple tabs**, each serving a specific function:

### **1️⃣ Data Preview**
- Upload **CSV** or **Excel** files.
- View the first few rows of the dataset. (head() in Pandas/Polars)
- Automatic data type detection. (dtypes in Pandas/Polars)
- Summary statistics & column insights. (describe() in Pandas/Polars)
- We use Polars by default for data loading and fall back to Pandas if needed.

### **2️⃣ Data Cleaning**
- Drop **NaN values**: Removes rows with missing data to avoid errors during analysis. (dropna() in Pandas / drop_nulls() in Polars)
- Remove **duplicate rows**: Ensures each record is unique for accurate analysis. (drop_duplicates() in Pandas / unique() in Polars)
- Change column data types: Converts columns to appropriate types (int, float, string) for proper computations. (astype() in Pandas / cast() in Polars)
- Normalize data: Rescales numerical features for consistent range.
	- Min-Max Scaling: (MinMaxScaler() from scikit-learn)
	- Z-Score: (StandardScaler() from scikit-learn)
- Apply row-based filtering: Keep rows based on conditions (e.g., value thresholds). (query() in Pandas / filter() in Polars)

### **3️⃣ Data Profiling**
- Generate comprehensive or sample-based data profiles.
- Powered by **YData Profiling**. Provides provision for data profiling using limited rows to improve speed.
- View correlations, missing values, distributions, and warnings.
- Functions used: ProfileReport() from ydata_profiling

### **4️⃣ Download & Summary**
- Download the processed dataset as **CSV**. (to_csv() in Pandas / write_csv() in Polars)
- See summary statistics of the cleaned dataset:
  - Column types

### **5️⃣ ML Prediction Module**
- Choose target and feature columns. The model is presently tested to use 'categorical' target class.
- Train a **Random Forest Classifier** on the uploaded dataset. (RandomForestClassifier() from scikit-learn)
- Dataset demonstrated is that of a 'balance scale measurements' obtained from OpenML.
- User can enter custom inputs for instant predictions post training.
- Tested for classification datasets. Regression model can be tested and implemented as 'Rain Forest' is capable of dealing with numerical classes as well. (extendable)
- Other Functions used: fit(), predict(), score()



---

## ⚙️ Stack Choices: Pandas vs Polars

This app supports **both Pandas and Polars** for data handling.

| Feature                | **Pandas** | **Polars** |
|------------------------|------------|------------|
| Performance            | Great for small/medium datasets | Optimized for large datasets & parallel processing |
| Memory Usage           | Higher     | Lower (more efficient memory allocation. It allows lazy loading.) |
| Syntax Compatibility   | Widely used, mature ecosystem | Similar to Pandas, growing rapidly |
| File Handling          | Adequate   | Faster CSV/Parquet read-write for big files |
| Multithreading         | Limited    | Built-in parallel execution |

**Why Polars?**
- Handles **large file sizes** efficiently.
- Offers **significant speed improvements** for reading, filtering, and aggregations.
- Uses **lazy evaluation** to optimize query execution.

In this app, for file loading:
- **Default:** Polars (for speed & performance).
- **Optional:** Switch to Pandas when needed.

In this app, for data manipulation:
- **Default:** Pandas (for simplicity & compatibility).
- **Optional:** Switch to Polars for large datasets (e.g., >50MB) for faster performance.

---


## 📊 Machine Learning Model: Random Forest

- Random Forests are **ensemble models**, which uses individual learners and then combines their learning to a single decision. It is often preferred as it is robust and adaptable. But due to cost intensive (High run time for larger data sets) and Black Box nature, it's often not advocated. 

- Black Box: Random Forests does not allow control on what the model does beyond a few hyper-parameters (Eg: Number of trees, depth etc). Thus it's very difficult to say why certain trees performed better while given higher weights.

![Random Forest](images/random_forest.png)

- Random forest builds multiple **decision trees** and merges them together to get accurate prediction. A large number of uncorrelated trees operate together to outperform individual models. Thus a forest is built with an ensemble of decision trees, usually trained with the "bagging method"

- The data used for training and testing in saved as 'balance_scale.csv'. Below is the data sample which depicts left-weight, left-distance, right-weight, right-distance.

![Random Forest Data Sample](images/datasample_randomforest.png)

- During training, the dataset is split into training and test sets (typically 70–80% for training, 20–30% for testing). The Random Forest model learns patterns by building multiple decision trees on random subsets of features and samples, which helps reduce overfitting and improves generalization.

- Each tree predicts the class independently, and the final prediction is made via majority voting across all trees in the forest. This ensemble approach ensures that individual tree errors are minimized.

- Hyperparameters such as the number of trees, maximum tree depth, and minimum samples per leaf are tuned to balance accuracy and computation time.

- Once trained, the model can predict the tipping direction for new scale configurations.


---


## ⚙️ Tips to handle large files

- **Already Implemented:**

	- Polars for Data Loading: Default data loader is Polars, which is faster and more memory-efficient than Pandas for large datasets.

	- Sample-Based Profiling: For very large datasets, only a subset of rows is used for profiling to improve speed and reduce memory usage.

	- Fallback to Pandas: If Polars cannot read a file (e.g., complex Excel sheets), Pandas is used as a fallback.

- **Possible Additions:**
	- Chunked Processing: Load data in chunks to handle extremely large files without memory issues.

	- Vectorized Operations: Use fully vectorized operations to speed up transformations and aggregations.


---

## 🖼️ Screenshots

---

## Data Loading

![Data Loading Screenshot](images/data_loading.png)

---

## Data Preview

![Data Preview Screenshot](images/data_preview.png)

---

## Data Cleaning

![Data Cleaning Screenshot](images/data_cleaning.png)

---

## Data Profiling

![Data Profiling Screenshot](images/data_profiling.png)

---

## Data Download & Summary

![Data Download Screenshot](images/data_download_summary.png)

---

## ML Prediction

![ML Prediction Screenshot](images/ml_prediction.png)


---

## 🛠️ Tech Stack

| Technology        | Purpose |
|-------------------|---------|
| **Python 3.10+**  | Core programming |
| **Streamlit**     | Interactive web UI |
| **Pandas / Polars** | Data handling |
| **scikit-learn**  | Machine Learning |
| **ydata-profiling** | Automated profiling |

---

## References:

- Machine Learning Model: Random Forest --> Learning links: 
		
	- https://www.blastanalytics.com/blog/comparing-propensity-modeling-techniques-to-predict-customer-behavior
	- https://swethadhanasekar.medium.com/random-forest-classifier-a-beginners-guide-c0b41713020

- Learning Random Forest Functions, Entropy, Gini Impurity using Microsoft Copilot

- Machine Learning Dataset: https://www.openml.org/search?type=data&sort=runs&id=11&status=active

- Used CHATGPT for learning about the user-interface of streamlit and programming guide

---

##App Hosted (URL):

App hosted live in https://asifnoushad-datarefinerapptool.streamlit.app/
