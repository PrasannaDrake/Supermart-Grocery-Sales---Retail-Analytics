# 🛒 Supermart Grocery Sales - Retail Analytics

This project analyzes sales data from a Supermart retail chain to uncover insights, trends, and patterns. It includes data cleaning, EDA, feature engineering, and a predictive model to forecast sales based on historical data.

---

## 📌 Project Highlights

- 🧼 Cleaned and preprocessed 8+ categorical and numerical columns
- 📊 Conducted EDA using bar plots, pie charts, and heatmaps
- 📈 Monthly sales trends across multiple years visualized
- 💡 Extracted custom insights like **Top 5 Cities by Avg Profit Margin**
- 🤖 Built a **Linear Regression model** to predict `Sales` from features
- 📉 Evaluated using **MSE** and **R² Score**

---

## 📁 Dataset

- **File**: `Supermart Grocery Sales - Retail Analytics Dataset.csv`
- **Fields Include**:
  - `Order Date`, `Sales`, `Profit`, `Discount`
  - `City`, `State`, `Region`
  - `Category`, `Sub Category`

---

## 🛠️ Tech Stack

- **Language**: Python 🐍
- **Libraries**:
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `sklearn`
  - `calendar` (for month mapping)

---

## 🔍 Exploratory Data Analysis

- 📌 Sales by Category  
- 📈 Monthly Trends  
- 🧭 Region-wise Sales Distribution  
- 💰 Profit Margin by City  
- 🔥 Correlation Heatmap for numerical attributes  

---

## 🧠 Model Training

```python
features = ['Category', 'Sub Category', 'City', 'Region', 'State', 'Month', 'Discount', 'Profit']
target = 'Sales'
