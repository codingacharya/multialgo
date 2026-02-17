import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Soil Health Dashboard", layout="wide")
st.title("🌱 Soil Health Dashboard (Optimized ML Version)")

st.markdown("Upload a CSV file with soil properties including Latitude & Longitude.")

# -------------------------------
# Upload File
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload Soil CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Sidebar Filters
    # -------------------------------
    st.sidebar.header("🔎 Filter Data")
    numeric_cols = df.select_dtypes(include=['float64','int64']).columns

    filtered_df = df.copy()
    for col in numeric_cols:
        min_val, max_val = float(df[col].min()), float(df[col].max())
        selected = st.sidebar.slider(col, min_val, max_val, (min_val, max_val))
        filtered_df = filtered_df[
            (filtered_df[col] >= selected[0]) &
            (filtered_df[col] <= selected[1])
        ]

    st.write(f"Filtered Samples: {filtered_df.shape[0]}")
    st.dataframe(filtered_df.head())

    # -------------------------------
    # Soil Health Score
    # -------------------------------
    st.markdown("### 🧮 Soil Health Score")

    health_cols = [col for col in
                   ['Soil_pH','Nitrogen','Phosphorus','Potassium']
                   if col in filtered_df.columns]

    if len(health_cols) >= 2:
        filtered_df['Health_Score'] = (
            filtered_df[health_cols] /
            filtered_df[health_cols].max()
        ).mean(axis=1) * 100
    else:
        filtered_df['Health_Score'] = 50

    # KPIs
    st.markdown("### 📌 KPIs")
    cols = st.columns(len(health_cols) + 1)

    for i, col in enumerate(health_cols):
        cols[i].metric(col, f"{filtered_df[col].mean():.2f}")

    cols[-1].metric("Health Score", f"{filtered_df['Health_Score'].mean():.2f}")

    # -------------------------------
    # Map
    # -------------------------------
    if 'Latitude' in filtered_df.columns and 'Longitude' in filtered_df.columns:
        st.markdown("### 🗺 Soil Map")

        m = folium.Map(
            location=[filtered_df['Latitude'].mean(),
                      filtered_df['Longitude'].mean()],
            zoom_start=8
        )

        for _, row in filtered_df.iterrows():
            score = row['Health_Score']
            color = "green" if score >= 75 else \
                    "orange" if score >= 50 else "red"

            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=6,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"Score: {score:.2f}"
            ).add_to(m)

        st_folium(m, width=700, height=500)

    # -------------------------------
    # Gauge
    # -------------------------------
    st.markdown("### 🏁 Health Score Gauge")

    if len(filtered_df) > 0:
        sample_id = st.selectbox("Select Sample", filtered_df.index)
        score = filtered_df.loc[sample_id, 'Health_Score']

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "Soil Health Score"},
            gauge={
                'axis': {'range': [0,100]},
                'steps': [
                    {'range': [0,50], 'color': "red"},
                    {'range': [50,75], 'color': "orange"},
                    {'range': [75,100], 'color': "green"}],
            }))

        st.plotly_chart(fig_gauge, use_container_width=True)

    # -------------------------------
    # Correlation
    # -------------------------------
    st.markdown("### 🔗 Correlation Heatmap")
    corr = filtered_df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # Histogram
    st.markdown("### 📈 Histogram")
    selected_col = st.selectbox("Select Column", numeric_cols)
    fig2, ax2 = plt.subplots()
    sns.histplot(filtered_df[selected_col], kde=True, ax=ax2)
    st.pyplot(fig2)

    # -------------------------------
    # Machine Learning
    # -------------------------------
    st.markdown("### 🤖 Machine Learning Prediction")

    ml_cols = [col for col in numeric_cols
               if col not in ['Latitude','Longitude']]

    target_col = st.selectbox("Select Target Variable", ml_cols)
    features = [col for col in ml_cols if col != target_col]

    X = filtered_df[features]
    y = filtered_df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20) / 100

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42
    )

    algorithm = st.selectbox(
        "Select Algorithm",
        ["Random Forest", "Linear Regression", "XGBoost", "KNN"]
    )

    if st.button("Train Model"):

        if algorithm == "Random Forest":
            model = RandomForestRegressor(
                n_estimators=300, max_depth=10, random_state=42)

        elif algorithm == "Linear Regression":
            model = LinearRegression()

        elif algorithm == "XGBoost":
            model = XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                random_state=42)

        elif algorithm == "KNN":
            model = KNeighborsRegressor(n_neighbors=7)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')

        st.markdown("### 📊 Model Performance")
        st.write(f"Algorithm: **{algorithm}**")
        st.write(f"RMSE: {rmse:.3f}")
        st.write(f"MAE: {mae:.3f}")
        st.write(f"R² Score (Test): {r2:.3f}")
        st.write(f"Cross-Validated R²: {cv_scores.mean():.3f}")

        if algorithm in ["Random Forest","XGBoost"]:
            st.markdown("### 🌟 Feature Importance")
            feat_imp = pd.DataFrame({
                "Feature": features,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)

            st.bar_chart(feat_imp.set_index("Feature"))

        fig3, ax3 = plt.subplots()
        sns.scatterplot(x=y_test, y=y_pred, ax=ax3)
        ax3.set_xlabel("Actual")
        ax3.set_ylabel("Predicted")
        st.pyplot(fig3)

        st.success("✅ Model Trained Successfully!")

    # -------------------------------
    # Predict New Sample
    # -------------------------------
    st.markdown("### 🔮 Predict New Sample")

    with st.expander("Enter Input Values"):
        input_data = {}

        for col in features:
            val = st.number_input(
                col,
                float(filtered_df[col].min()),
                float(filtered_df[col].max()),
                float(filtered_df[col].mean())
            )
            input_data[col] = val

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)
            st.success(f"Predicted {target_col}: {prediction[0]:.3f}")

else:
    st.info("👆 Upload a CSV file to begin.")
