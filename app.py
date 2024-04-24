import streamlit as st
import pandas as pd
from pycaret.classification import setup as classification_setup, compare_models as classification_compare_models
from pycaret.regression import setup as regression_setup, compare_models as regression_compare_models

def main():
    st.title("Welcome to my Project PyCaret :)")

    # Upload CSV file
    file = st.file_uploader("Please upload your CSV file", type=['csv'])

    if file is not None:
        # Load data
        data = pd.read_csv(file)
        st.dataframe(data.head())

        # Drop columns
        if st.checkbox("Drop columns"):
            columns_to_drop = st.multiselect("Select columns to drop", data.columns)
            data.drop(columns_to_drop, axis=1, inplace=True)
            st.write("Columns dropped successfully!")

        # Exploratory Data Analysis
        if st.checkbox("Perform Exploratory Data Analysis (EDA)"):
            selected_columns = st.multiselect("Select columns for EDA", data.columns)
            if selected_columns:
                st.write(data[selected_columns].describe())

                # Option to display histogram
                if st.checkbox("Show Histogram for selected columns"):
                    for column in selected_columns:
                        st.subheader(f"Histogram for {column}")
                        st.hist(data[column])

        # Handle missing values
        handle_missing = st.radio("How to handle missing values?", ('Drop rows', 'Impute'))
        if handle_missing == 'Drop rows':
            data.dropna(inplace=True)
            st.write("Missing values handled by dropping rows.")
        elif handle_missing == 'Impute':
            st.info("Imputation methods can be added here.")

        # Encode categorical data
        encode_categorical = st.radio("How to encode categorical data?", ('One Hot Encoding', 'Label Encoding'))
        if encode_categorical == 'One Hot Encoding':
            data = pd.get_dummies(data)
            st.write("Categorical data encoded using One Hot Encoding.")
        elif encode_categorical == 'Label Encoding':
            st.info("Label encoding methods can be added here.")

        # Choose X and y variables
        target_variable = st.selectbox("Select the target variable", data.columns)
        X = data.drop(columns=[target_variable], axis=1)  # Assuming target variable is known
        y = data[target_variable]

        # Detect task type
        task_type = 'classification' if y.dtype == 'object' else 'regression'

        # Train models
        st.subheader("Training Models")
        if task_type == 'classification':
            classification_setup(data, target=target_variable)
            best_model = classification_compare_models()
        elif task_type == 'regression':
            regression_setup(data, target=target_variable)
            best_model = regression_compare_models()

        st.subheader("Best Model")
        st.write(best_model)
        


if __name__ == "__main__":
    main()

 
