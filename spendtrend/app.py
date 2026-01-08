#  B01813057_NCL_COMP09118_CW
#  Imports
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error # Added r2_score and mean_absolute_error which was not in my original planning documentation

# FontAwesome CDN for UI icons
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
""", unsafe_allow_html=True)

# Page configuration
st.set_page_config(page_title="UK Household Spending Dashboard",
layout="wide")

# Page Title and Subtitle
st.markdown('<h1 class="main-header"><i class="fas fa-chart-line" style="font-size: 2.5rem;"></i> UK Household Spending Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**Analyzing spending trends from 2001-2024 with COVID impact analysis**")

# Sidebar for file upolad + filters
with st.sidebar:
    @st.dialog("User Guide")
    def user_guide():
        st.write("This is a user guide for the UK Household Spending Dashboard.")
        st.write("1. Upload a CSV dataset using the 'Browse Files' button.")
        st.write("2. Click the 'Analyse File' button underneath the raw dataset preview (if enabled).")
        st.write("3. Select a commodity from the dropdown menu (Total expenditure is the default commodity used for this project).")
        st.write("4. Ensure all filters you require are checked, and you have selected the years you want to predict (if applicable).")
        st.write("5. Click the 'Display Visualisation' button to see the visualisation of the data.")
        st.write("6. The visualisation will load and show the trend of the data you have selected as well as the predictions (if applicable).")
    
    if st.button("User Guide", use_container_width=True, type="primary"):
        user_guide()
    
    st.markdown("## Dashboard Filters")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["xlsx", "csv"], help="Upload your UK household spending data")

# If a file is uploaded, read the file and display a success message
    if uploaded_file is not None: 
        df = pd.read_csv(uploaded_file)
        st.write("File uploaded successfully!")
        df = df.dropna(axis=1, how='all')  # Drop any columns that are completely empty
        year_columns = [col for col in df.columns if "20" in str(col)]  # Get all columns that contain the string of "20"

#  Checkbox filters to show/hide the raw dataset preview, covid impact, trends, and predictions
        show_raw_dataset_preview = st.checkbox("Show Raw Dataset Preview", value=True)
        show_covid_impact = st.checkbox("Show COVID Impact", value=True)
        show_trends = st.checkbox("Show Trends", value=True)
        show_predictions = st.checkbox("Show Predictions", value=True)

#  If predictions are checked, give the uesr the option to select specific years to predict
        if show_predictions:
            years = st.multiselect("Years to predict",
            [2025, 2026, 2027, 2028, 2029, 2030],
            default = [2025],
            max_selections = 10,
            accept_new_options = True)

    if uploaded_file is None:  #  If there is no uploaded file, display an info message
        st.info("Please upload a file to begin analysis", icon="ℹ️")
        st.stop()

#  If there is an uploaded file, and the raw dataset preview is checked, display the first 5 rows of the dataset
if uploaded_file is not None and show_raw_dataset_preview:
    st.markdown("### Raw Dataset Preview")
    st.dataframe(df.head(5))

# -------- File Analysis Section --------
if st.button("Analyse File", type="primary"):  #  If the analyse file button is clicked, analyse the file
    with st.spinner("Analysing file..."):

        cols = []  #  Create an empty list to store the columns
        for c in df.columns:
            if '-' in str(c):  #  If the column contains a hyphen
                start, end = str(c).split('-')  #  Splits the column into start and end
                if int(start) < 2010:  #  If the start is less than 2010, set the year to the starting year
                    year = int(start)
                else:  #  If the start is greater than 2010, set the year to the ending year
                    end = "20" + end  #  Add the string of "20" to the start of the ending year (if the year is '18' it would become '2018')
                    year = int(end)
                year = int(start) if int(start) > int(end) else int(end)  
                cols.append(year)
            else:  #  If the column does not contain a hyphen, set the year to the column name
                cols.append(int(c) if str(c).isdigit() else c)  #  If the column is a digit, set the year to the column name

        df.columns = cols
        df = df.loc[:, ~df.columns.duplicated()]  #  Drop any columns that are duplicated
        df = df.dropna(axis=1, how='all')  #  Drop any rows that are completely empty
        df = df.dropna(axis=0, how='all')  #  Drop any columns that are completely empty
        df = df.drop(2006, axis=1)  #  Drop the column with the year 2006
        df = df.rename(columns={'2006.1': '2006'})  #  Rename the column with the year 2006.1 to 2006 (to match the weighting method used in the rest of teh data)
        st.session_state["cleaned_df"] = df
        st.write(df.head(5))  #  Display the first 5 rows of the cleaned dataset

if "cleaned_df" in st.session_state:  
    clean_df = st.session_state["cleaned_df"]
    commodities = clean_df['Commodity or service'].tolist()
    # Find index of "Total expenditure" or default to 0
    default_index = commodities.index("Total expenditure") if "Total expenditure" in commodities else 0
    selected_commodity = st.selectbox("Select a commodity", commodities, index=default_index)  #  Display the list of commodities in a dropdown menu
    row = clean_df[clean_df["Commodity or service"] == selected_commodity]  #  Get the row for the selected commodity
    clean_df.columns = [int(c) if str(c).isdigit() else c for c in clean_df.columns]  #  Convert the columns to integers if they are digits, otherwise keep them as strings
    st.write(row)  
    st.write("Selected commodity: ", selected_commodity)  

    if st.button("Display Visualisation", type="primary"):  #  If the Display Visualisation button is clicked, display the visualisation of the data to gain insigtt
        with st.spinner("Displaying visualisation..."):

            plt.rcParams["savefig.facecolor"] = "White"  #  Set the background color of the figure to white
            plt.rcParams["axes.facecolor"] = "White"  #  Set the background color of the axes to white
            year_columns = [col for col in clean_df.columns if isinstance (col, int)]  #  Get all the columns that are integers
            x = year_columns  #  Set the x-axis to the year columns
            y = pd.to_numeric(row[year_columns].iloc[0], errors="coerce").values  #  Convert the value to numeric, and any errors should be coerced

            x = np.array(year_columns)  #  Convert the year columns to an np.array
            mask = ~np.isnan(y)  #  Create mask to remove any NaN values
            x_full = x[mask].reshape(-1, 1)  #  Full x-axis (including COVID years)
            y_full = y[mask]  #  Full y-axis (including COVID years)

            # Remove COVID years from regression if box is checked (but keep them on the graph)
            if show_covid_impact:
                covid_mask = ~((x_full.flatten() >= 2020) & (x_full.flatten() <= 2022))
                x_train = x_full[covid_mask].reshape(-1, 1)  #  Training data (COVID removed)
                y_train = y_full[covid_mask]
            else:
                x_train = x_full
                y_train = y_full

            from sklearn.model_selection import train_test_split  #  Train/test split

            x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(
                x_train, y_train, test_size=0.2, random_state=42
            )

            model = LinearRegression().fit(x_train_split, y_train_split)  #  Train the model using the training data

            r2_train = r2_score(y_train_split, model.predict(x_train_split))  #  Training r2 score
            r2_test = r2_score(y_test_split, model.predict(x_test_split))  #  Testing r2 score (unseen data)
            st.write(f"**Training R² Score:** {r2_train:.4f}")
            st.write(f"**Testing R² Score (Unseen Data):** {r2_test:.4f}")

            mae_train = mean_absolute_error(y_train_split, model.predict(x_train_split))  #Training MAE score
            mae_test = mean_absolute_error(y_test_split, model.predict(x_test_split))  #Testing MAE score
            st.write(f"**Mean Absolute Error (Training):** {mae_train:.4f}")
            st.write(f"**Mean Absolute Error (Testing):** {mae_test:.4f}")


            fig, ax = plt.subplots(figsize=(10, 5))  #  Create a figure and axis

            if show_covid_impact:
                ax.axvspan(2020, 2022, color="red", alpha=0.25, label="COVID-19 Period (2020-2022)")  #  Add a red shaded area to the graph to represent the COVID-19 period if checked
                # Plot all data points (including COVID years) if checked
                ax.scatter(x_full, y_full, color ="black", label="COVID Data")  #  As all data points have their own colours, this is for COVID data only

            # Plot training vs testing points separately
            ax.scatter(x_train_split, y_train_split, color="blue", label="Training Data")  #  Plot the training data
            ax.scatter(x_test_split, y_test_split, color="green", label="Testing Data")  #  Plot the testing data

            # Regression line MUST use full x range for smoother prediction
            if show_trends:
                ax.plot(x_full, model.predict(x_full), label="Regression Line")  #  Plot the regression line if checked

            if show_predictions and years:  #  If predictions and years are selected, plot the predicted data
                future_x = np.array(years).reshape(-1, 1)  #  Convert the years to an np.array
                future_predictions = model.predict(future_x)  #  Predict the future data
                
                # Get the last historical year and its prediction to connect the lines
                last_historical_year = x_full[-1].flatten()[0]  # Extract the last year from the x_full array
                last_historical_prediction = model.predict(x_full[-1].reshape(-1, 1))[0]  # Predict the last year from the x_full array
                
                # Combine last historical point with future predictions for a continued regression line from 2024 onto the prediction line
                connected_x = np.concatenate([[last_historical_year], future_x.flatten()])
                connected_y = np.concatenate([[last_historical_prediction], future_predictions])

                ax.scatter(future_x, future_predictions, color="purple", label="Predicted")  #  Plot the predicted data
                ax.plot(connected_x, connected_y, color="purple", linestyle="--")  #  Plot the predicted data connected from 2024

                for x_val, y_val in zip(future_x, future_predictions):
                    offset = y_val * 0.02
                    ax.text(x_val, y_val - offset, f"£{y_val:.2f}", fontsize=8, ha="center", va="top")

            ax.set_xlabel("Year")
            ax.set_ylabel("Average Weekly Expenditure (£)")
            ax.set_title(f"Trend for {selected_commodity}")
            ax.legend()
            st.pyplot(fig)




















