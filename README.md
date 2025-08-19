# 📊 Data Refiner App

An interactive **data cleaning, profiling, and prediction platform** built with **Streamlit**.  
We can quickly upload, explore, clean, and run machine learning predictions from a browser-based interface.

---

## 🚀 Features

The app is organized into **multiple tabs**, each serving a specific function:

<details>
  <summary><b>1️⃣ Data Loading & Preview</b></summary>

- Upload **CSV** or **Excel** files.
- Load CSV files via lazy loading feature of pandas and display 5 rows as preview.
- Load entire dataset for further cleaning process.
- We use Polars by default for data loading and fall back to Pandas if needed.
- View the first 100 rows of the dataset. (head() in Pandas/Polars)
- Automatic data type detection. (dtypes in Pandas/Polars)

![Data Loading Screenshot](images/data_loading.png)

![Data Preview Screenshot](images/data_preview.png)

</details>

---

<details>
  <summary><b>2️⃣ Data Cleaning</b></summary>

- Drop **NaN values**: Removes rows with missing data to avoid errors during analysis.
- Remove **duplicate rows**.
- **Type Conversion**: Converts columns to appropriate types.
- **Normalize data**:
  - **Min-Max Scaling**  
    Equation: `(Value - Min)/(Max - Min)`
  - **Z-Score Normalization**  
    Equation: `(Value - Mean)/Std`

- Apply row-based filtering (query/filter).

![Data Cleaning Screenshot](images/data_cleaning.png)

</details>

---

<details>
  <summary><b>3️⃣ Data Profiling</b></summary>

- Generate comprehensive or sample-based data profiles.
- By default loads 1000 random sampled rows.
- Powered by **YData Profiling**.
- Functions used: `ProfileReport()`.

![Data Profiling Screenshot](images/data_profiling.png)

</details>

---

<details>
  <summary><b>4️⃣ Download & Summary</b></summary>

- Download the processed dataset as **CSV**.  
- See summary statistics of the cleaned dataset.

![Data Download Screenshot](images/data_download_summary.png)

</details>

---

<details>
  <summary><b>5️⃣ ML Prediction Module</b></summary>

- Choose target and feature columns.
- Train a **Random Forest Classifier**.  
- User can enter custom inputs for predictions.  
- Dataset: Balance scale measurements from OpenML.  

![ML Prediction Screenshot](images/ml_prediction.png)

</details>

---

## 📊 Machine Learning Model: Random Forest

<details>
  <summary><b>Click to expand Random Forest details</b></summary>

- Random Forests are **ensemble models**, combining multiple decision trees.  
- Advantage: robust, reduces overfitting.  
- Drawback: costly for large datasets & acts as a black box.

![Random Forest](images/random_forest.png)

- Training dataset: `balance_scale.csv` with left-weight, left-distance, right-weight, right-distance.

![Random Forest Data Sample](images/datasample_randomforest.png)

- Dataset split into train/test.  
- Predictions based on majority voting across trees.  
- Hyperparameters: n_estimators, max_depth, min_samples_leaf, etc.  

</details>

---

## ⚙️ Stack Choices: Pandas vs Polars

<details>
  <summary><b>Click to expand Pandas vs Polars comparison</b></summary>

| Feature                | **Pandas** | **Polars** |
|------------------------|------------|------------|
| Performance            | Great for small/medium datasets | Optimized for large datasets & parallel processing |
| Memory Usage           | Higher     | Lower |
| Syntax Compatibility   | Mature ecosystem | Similar to Pandas, growing rapidly |
| File Handling          | Adequate   | Faster CSV/Parquet read-write |
| Multithreading         | Limited    | Built-in parallel execution |

**Why Polars?**
- Handles **large file sizes** efficiently.
- **Lazy evaluation** optimizes queries.

**When does it shift to Pandas?**
- Non-UTF8 encodings.  
- Inconsistent data schema.  

</details>

---

## ⚙️ Tips to handle large files (Optimizations)

<details>
  <summary><b>Click to expand Optimizations</b></summary>

**Already Implemented:**
- Polars default for data loading.  
- Fallback to Pandas if needed.  
- Lazy Loading for Display (first 5 rows).  
- Sample-Based Profiling (1000 rows).  
- Selective profiling to reduce overhead.  

**Possible Additions:**
- Caching.  
- Chunked processing.  
- Vectorized operations.  

</details>

---

## 🛠️ Tech Stack

<details>
  <summary><b>Click to expand Tech Stack</b></summary>

| Technology        | Purpose |
|-------------------|---------|
| **Python 3.10+**  | Core programming |
| **Streamlit**     | Interactive web UI |
| **Pandas / Polars** | Data handling |
| **scikit-learn**  | Machine Learning |
| **ydata-profiling** | Automated profiling |

</details>

---

## 📚 References

<details>
  <summary><b>Click to expand References</b></summary>

- Random Forest articles:  
  - https://www.blastanalytics.com/blog/comparing-propensity-modeling-techniques-to-predict-customer-behavior  
  - https://swethadhanasekar.medium.com/random-forest-classifier-a-beginners-guide-c0b41713020  

- Dataset: https://www.openml.org/search?type=data&sort=runs&id=11&status=active  

- Used ChatGPT for UI & coding guidance.  

</details>

---

## 🌐 App Hosted

App live 👉 [Streamlit Deployment](https://asifnoushad-datarefinerapptool.streamlit.app/)

