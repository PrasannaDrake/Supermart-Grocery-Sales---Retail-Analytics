# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load Dataset (use raw string or forward slashes)
df = pd.read_csv(r"Project3_SupermarketRetailAnalytics/Supermart Grocery Sales - Retail Analytics Dataset.csv")

# Step 3: Preprocessing - Handle Duplicates and Dates
df.drop_duplicates(inplace=True)
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')

# Drop rows with invalid dates or essential NaNs
df.dropna(subset=['Order Date', 'Sales', 'Profit'], inplace=True)

# Extract month and year
df['Month'] = df['Order Date'].dt.month
df['Year'] = df['Order Date'].dt.year

# Safe Month Name Mapping
df['Month_Name'] = df['Month'].apply(lambda x: calendar.month_name[int(x)] if pd.notna(x) and 1 <= x <= 12 else 'Unknown')

# Step 4: Encode Categorical Variables
le = LabelEncoder()
for col in ['Category', 'Sub Category', 'City', 'Region', 'State']:
    df[col] = le.fit_transform(df[col].astype(str))

# Step 5: Exploratory Data Analysis (EDA)
# --- Category-wise Sales ---
plt.figure(figsize=(10,6))
sns.barplot(data=df.groupby('Category')['Sales'].sum().reset_index(), x='Category', y='Sales')
plt.title('Sales by Category')
plt.show()

# --- Monthly Sales Trend ---
monthly_sales = df.groupby(['Year', 'Month_Name'])['Sales'].sum().reset_index()
plt.figure(figsize=(12,6))
sns.lineplot(data=monthly_sales, x='Month_Name', y='Sales', hue='Year')
plt.title('Monthly Sales Trend')
plt.xticks(rotation=45)
plt.show()

# --- Correlation Heatmap ---
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='YlGnBu')
plt.title("Correlation Matrix")
plt.show()

# Step 6: Custom Insight - Top 5 Cities by Profit Margin
df = df[df['Sales'] != 0]  # Avoid division by zero
df['Profit Margin %'] = (df['Profit'] / df['Sales']) * 100
top_cities = df.groupby('City')['Profit Margin %'].mean().sort_values(ascending=False).head(5)

top_cities.plot(kind='bar', figsize=(10,5), color='teal')
plt.title("Top 5 Cities by Avg Profit Margin (%)")
plt.ylabel("Profit Margin %")
plt.show()

# --- Sales Contribution by Region ---
region_contrib = df.groupby('Region')['Sales'].sum()
region_contrib.plot(kind='pie', autopct='%1.1f%%', figsize=(8,8))
plt.title("Sales Distribution by Region")
plt.ylabel('')
plt.show()

# Step 7: Feature Engineering for Modeling
features = df[['Category', 'Sub Category', 'City', 'Region', 'State', 'Month', 'Discount', 'Profit']]
target = df['Sales']

# Fill any remaining NaNs in features
features = features.fillna(0)

# Step 8: Feature Scaling
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Step 9: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Step 10: Model Training and Evaluation
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Step 11: Actual vs Predicted Sales Visualization
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='purple')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
