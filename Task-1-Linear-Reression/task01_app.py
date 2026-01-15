import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="House Price Prediction", layout="wide")

st.title("🏠 House Price Prediction Dashboard")

# Load data
data = pd.read_csv("train.csv")
df = data[["GrLivArea", "BedroomAbvGr", "FullBath", "SalePrice"]].dropna()

X = df[["GrLivArea", "BedroomAbvGr", "FullBath"]]
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse=mse**0.5

# ---- UI ----
col1, col2, col3 = st.columns(3)

col1.metric("Estimated Price (sample)", f"${int(y_pred[0]):,}")
col2.metric("R² Score", round(r2, 3))
col3.metric("RMSE", f"${int(rmse):,}")

st.subheader("🔍 Enter house details")

sqft = st.slider("Square Footage", 500, 4000, 1500)
bed = st.slider("Bedrooms", 1, 6, 3)
bath = st.slider("Bathrooms", 1, 4, 2)

if st.button("Predict Price"):
    pred = model.predict([[sqft, bed, bath]])
    st.success(f"Estimated House Price: ${int(pred[0]):,}")
