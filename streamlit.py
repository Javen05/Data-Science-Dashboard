# Import required libraries
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
st.set_page_config(layout="wide", initial_sidebar_state="auto")

# Load the dataset
dataset = 'Assignment2_Dataset.csv'
df = pd.read_csv(dataset, encoding='cp1252', parse_dates=['Date'], dayfirst=False, index_col='S/N')

# Data Preprocessing
df.drop([3533, 2610, 8385], inplace=True)
df = df[(df['Rented_Bike_Count'] != -2) & (df['Rented_Bike_Count'] != 34180.0)]
df = df.dropna(subset=['Rented_Bike_Count'])
df['Hour'].interpolate(method='linear', inplace=True)  # Fill missing values in Hour using linear interpolation
df['Temperature'].interpolate(method='quadratic', inplace=True)
df['Temperature'] = df['Temperature'].round(decimals=1)  # Round off the interpolated values to one decimal place
df = df[(df['Temperature'] >= -41) & (df['Temperature'] <= 41)]

def calculate_humidity(temperature, dew_point):
    # formula
    numerator = np.exp((17.625 * dew_point) / (243.04 + dew_point))
    denominator = np.exp((17.625 * temperature) / (243.04 + temperature))
    relative_humidity = 100 * (numerator / denominator)

    # round to nearest whole number
    return np.round(relative_humidity)

# Impute rows with missing Humidity
df.loc[df['Humidity'].isnull(), 'Humidity'] = calculate_humidity(df['Temperature'], df['Dewpoint_Temp'])

df['Snowfall'].fillna('no_snowfall', inplace=True)
df = df.dropna(subset=['Hit_Sales'])
df['Open'] = df['Open'].replace('n', 'No')
df['Open'] = df['Open'].replace(['Y', 'yes', 'yes '], 'Yes')
df['Snowfall'] = df['Snowfall'].replace({'no_snowfall': 0, 'low': 1, 'medium': 2, 'heavy': 3, 'very heavy': 4})
df['Hit_Sales'] = df['Hit_Sales'].replace({'N': 0, 'Y': 1})
df['Open'] = df['Open'].replace({'No': 0, 'Yes': 1})

# Define the bin ranges and interpretive labels for each attribute
bin_ranges = {
    'Rented_Bike_Count': [-1, 0, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500],
    'Temperature': [-30, -20, -10, 0, 10, 20, 30, 41],
    'Windspeed': [-1, 0.9, 2, 3, 5, 7.5],
    'Dewpoint_Temp': [-40, -20, 0, 20, 41],
    'Solar_Radiation': [-1, 0.01, 0.5, 1, 2, 4],
    'Rainfall': [-1, 5, 10, 20, 30, 36]
}

interpretive_labels = {
    'Rented_Bike_Count': ['No rentals', '1-100', '101-500', '501-1000', '1001-1500', '1501-2000', '2001-2500', '2501-3000', '3001-3500'],
    'Temperature': ['-30 to -20', '-20 to -10', '-10 to 0', '0 to 10', '10 to 20', '20 to 30', '30+'],
    'Windspeed': ['0-0.9', '0.9-2', '2-3', '3-5', '5+'],
    'Dewpoint_Temp': ['<-20', '-20 to 0', '0 to 20', '20+'],
    'Solar_Radiation': ['0', '0.01-0.5', '0.5-1', '1-2', '2+'],
    'Rainfall': ['No rain', '5-10', '10-20', '20-30', '30+']
}

# Perform binning and label assignment for each attribute
for attribute, bins in bin_ranges.items():
    df[f'{attribute}_Bins'] = pd.cut(df[attribute], bins=bins, labels=interpretive_labels[attribute])

def comfort_scale(row):
    score = 0
    if 20 <= row['Temperature'] <= 30:
        score += 1
    if 30 <= row['Humidity'] <= 60:
        score += 1
    if row['Windspeed'] <= 3:
        score += 1
    if row['Visibility'] <= 3:
        score += 1
    if row['Rainfall'] == 0:
        score += 1
    return score

df['Comfort_Scale'] = df.apply(comfort_scale, axis=1)

# Create new Columns
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

month_mapping = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
}

df['Month'] = df['Month'].map(month_mapping)
df['Day'] = df['Date'].dt.day_name()

df = df[df['Rented_Bike_Count'] > 0]

months = [['Dec', 'Jan', 'Feb'], ['Mar', 'Nov', 'Apr'], ['Oct', 'Aug', 'Sep', 'May', 'Jul'], 'Jun']

month_to_encoded = {}
encoded_month = 1

for month in months:
    if isinstance(month, list):
        for m in month:
            month_to_encoded[m] = encoded_month
        encoded_month += 1
    else:
        month_to_encoded[month] = encoded_month
        encoded_month += 1

df['Encoded_Month'] = df['Month'].map(month_to_encoded)

hours = [[4, 5], [0, 1, 2, 3], [6, 7], [10, 11], 9, [12, 13], [14, 15, 16], [20, 21, 22, 23], 8, [17, 18, 19]]

hour_to_encoded = {}
encoded_hour = 0

for hour in hours:
    if isinstance(hour, list):
        for h in hour:
            hour_to_encoded[h] = encoded_hour
        encoded_hour += 1
    else:
        hour_to_encoded[hour] = encoded_hour
        encoded_hour += 1

df['Encoded_Hour'] = df['Hour'].map(hour_to_encoded)

def train_knn_model():
    df2 = df[df['Rented_Bike_Count'] > 0] # don't include observations where stores are closed.
    FEATURES = ['Temperature', 'Humidity', 'Solar_Radiation', 'Rainfall', 'Encoded_Hour', 'Encoded_Month', 'Comfort_Scale']
    BIKE = ['Rented_Bike_Count']

    Y = df2[BIKE]
    X = df2[FEATURES]

    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X, Y)

    return knn_model

knn_model = train_knn_model()

FEATURES = ['Temperature', 'Humidity', 'Solar_Radiation', 'Rainfall', 'Encoded_Hour', 'Encoded_Month', 'Comfort_Scale']
HIT_SALES = ['Hit_Sales']       # classification
BIKE = ['Rented_Bike_Count']    # regression
Y = df[BIKE]               # set target
X = df[FEATURES]           # set x predictors

# Train the Random Forest model
def train_random_forest_model():
    df2 = df[df['Rented_Bike_Count'] > 0] # don't include observations where stores are closed.
    RF_FEATURES = ['Temperature', 'Encoded_Hour', 'Humidity', 'Solar_Radiation', 'Encoded_Month', 'Rainfall', 'Comfort_Scale']
    HIT_SALES = ['Hit_Sales']

    Y = df2[HIT_SALES]
    X = df2[RF_FEATURES]

    random_forest_model = RandomForestClassifier(n_estimators=100, random_state=2202934)
    random_forest_model.fit(X, Y.values.ravel())

    return random_forest_model

random_forest_model = train_random_forest_model()

# Create Streamlit web app
st.title("Ohaiyo Business Analytics Dashboard")

# User input section
st.sidebar.header("Predictor Parameters")
temperature = st.sidebar.slider("Temperature (°C)", -30.0, 41.0, 20.0)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
solar_radiation = st.sidebar.slider("Solar Radiation (MJ/m^2)", -5.0, 5.0, 0.1)  # Adjust step to 0.1
rainfall = st.sidebar.slider("Rainfall (mm)", 0, 36, 0)
Hour = st.sidebar.slider("Time (24-Hour format)", 0, 23, 12)
Month = st.sidebar.selectbox("Month", ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
windspeed = st.sidebar.radio("Is it windy?", ['Yes', 'No'])
visibility = st.sidebar.radio("Is weather expected to be clear?", ['Yes', 'No'])

# Encoding for the models to understand
# calculating comfort scale
Comfort_Scale = 0

if temperature >= 20 and temperature <= 30:
    Comfort_Scale += 1

if humidity >= 30 and humidity <= 60:
    Comfort_Scale += 1

if windspeed == 'No':
    Comfort_Scale += 1

if visibility == 'Yes':
    Comfort_Scale += 1

if rainfall == 0:
    Comfort_Scale += 1
    
    
# ranking month
months = [['Dec', 'Jan', 'Feb'], ['Mar', 'Nov', 'Apr'], ['Oct', 'Aug', 'Sep', 'May', 'Jul'], 'Jun']
month_to_encoded = {}
encoded_month = 1

for month in months:
    if isinstance(month, list):
        for m in month:
            month_to_encoded[m] = encoded_month
        encoded_month += 1
    else:
        month_to_encoded[month] = encoded_month
        encoded_month += 1
        
Encoded_Month = month_to_encoded.get(Month, -1)


# ranking hour
hours = [[4,5], [0,1,2,3], [6,7], [10,11], 9, [12,13], [14,15,16], [20,21,22,23], 8, [17, 18, 19]]
hour_to_encoded = {}
encoded_hour = 0

for hour in hours:
    if isinstance(hour, list):
        for h in hour:
            hour_to_encoded[h] = encoded_hour
        encoded_hour += 1
    else:
        hour_to_encoded[hour] = encoded_hour
        encoded_hour += 1

Encoded_Hour = hour_to_encoded.get(Hour, -1)


# formatting parameters for models
knn_data = pd.DataFrame({
    'Temperature': [temperature],
    'Humidity': [humidity],
    'Solar_Radiation': [solar_radiation],
    'Rainfall': [rainfall],
    'Encoded_Hour': [Encoded_Hour],
    'Encoded_Month': [Encoded_Month],
    'Comfort_Scale': [Comfort_Scale]
})

rf_data = pd.DataFrame({
    'Temperature': [temperature], 
    'Encoded_Hour': [Encoded_Hour],
    'Humidity': [humidity],
    'Solar_Radiation': [solar_radiation],
    'Encoded_Month': [Encoded_Month],
    'Rainfall': [rainfall],
    'Comfort_Scale': [Comfort_Scale]
})

# Predict bike rentals using the K-nearest neighbors model
predicted_rentals = knn_model.predict(knn_data)

# Predict sales performance using the Random Forest model
predicted_sales = random_forest_model.predict(rf_data)

# Display predictions
st.header("Performance Prediction")
if len(predicted_rentals) > 0:
    st.write(f"You should expect around {round(predicted_rentals[0][0])} bicycle rentals.")
else:
    st.write("No prediction available.")
st.write(f"Hit sales is {'EXPECTED' if predicted_sales[0] == 1 else 'NOT EXPECTED'} to be achieved.")

st.write(temperature, humidity, solar_radiation, rainfall, Encoded_Month, Encoded_Hour, Comfort_Scale)


dfN = pd.read_csv(dataset, encoding='cp1252', parse_dates=['Date'], dayfirst=False, index_col='S/N')

# User input section
st.sidebar.header("Enter New Record")

# Add input fields for the new record
new_record_date = st.sidebar.date_input("Date")
new_record_country = st.sidebar.selectbox("Country", ['JP', 'SK'])
new_record_rented_bike_count = st.sidebar.number_input("Rented Bike Count", min_value=0)
new_record_hour = st.sidebar.number_input("Hour", min_value=0, max_value=23)
new_record_temperature = st.sidebar.number_input("Temperature (°C)", value=20.0)
new_record_humidity = st.sidebar.number_input("Humidity (%)", value=50)
new_record_windspeed = st.sidebar.number_input("Windspeed", value=0.0)
new_record_visibility = st.sidebar.number_input("Visibility", value=3.0)
new_record_dewpoint_temp = st.sidebar.number_input("Dewpoint Temperature", value=-4.8)
new_record_solar_radiation = st.sidebar.number_input("Solar Radiation", value=0.0)
new_record_rainfall = st.sidebar.number_input("Rainfall", value=0.0)
new_record_snowfall = st.sidebar.radio("Snowfall", ['no_snowfall', 'low', 'medium', 'heavy', 'very heavy'])
new_record_open = st.sidebar.radio("Open", ['Yes', 'No'])
new_record_hit_sales = st.sidebar.radio("Hit Sales", ['N', 'Y'])
new_record_region = st.sidebar.text_input("Region")
new_record_latitude = st.sidebar.number_input("Latitude")
new_record_longitude = st.sidebar.number_input("Longitude")

# Add a button to add the new record to the CSV file
if st.sidebar.button("Add New Record"):
    # Perform data sanitization and validation
    if (new_record_date is not None and
        new_record_country and
        new_record_rented_bike_count >= 0 and
        0 <= new_record_hour <= 23 and
        -41 <= new_record_temperature <= 41 and
        0 <= new_record_humidity <= 100 and
        0 <= new_record_windspeed and
        0 <= new_record_visibility and
        -41 <= new_record_dewpoint_temp <= 41 and
        0 <= new_record_solar_radiation and
        0 <= new_record_rainfall and
        new_record_snowfall in ['no_snowfall', 'low', 'medium', 'heavy', 'very heavy'] and
        new_record_open in ['Yes', 'No'] and
        new_record_hit_sales in ['N', 'Y'] and
        new_record_region and
        -90 <= new_record_latitude <= 90 and
        -180 <= new_record_longitude <= 180):
        # Create a new record as a dictionary
        new_record = {
            'Date': new_record_date,
            'Country': new_record_country,
            'Rented_Bike_Count': new_record_rented_bike_count,
            'Hour': new_record_hour,
            'Temperature': new_record_temperature,
            'Humidity': new_record_humidity,
            'Windspeed': new_record_windspeed,
            'Visibility': new_record_visibility,
            'Dewpoint_Temp': new_record_dewpoint_temp,
            'Solar_Radiation': new_record_solar_radiation,
            'Rainfall': new_record_rainfall,
            'Snowfall': new_record_snowfall,
            'Open': new_record_open,
            'Hit_Sales': new_record_hit_sales,
            'Region': new_record_region,
            'Latitude': new_record_latitude,
            'Longitude': new_record_longitude
        }

        # Append the new record to the DataFrame
        dfN = dfN.append(new_record, ignore_index=True)

        # Save the updated DataFrame to the CSV file
        dfN.to_csv(dataset, encoding='cp1252', index_label='S/N')

        # Display a success message
        st.success("New record added successfully!")

    else:
        # Display an error message for invalid input
        st.error("Invalid input. Please check the values and try again.")

# Display the dataset
st.write("### Historical Records")
st.write(dfN.iloc[::-1])


st.header("Performance Visualization")
import matplotlib.pyplot as plt
import seaborn as sns

# Visualization 1: Group by hit_sales, and find the mean rented_bike_count for Yes and No
average_rented_count = df.groupby('Hit_Sales')['Rented_Bike_Count'].mean()

# Visualization 2: Ratio of Hit Sales vs. No Hit Sales
ratio_hit_sales = df['Hit_Sales'].value_counts(normalize=True)

# Create a two-column layout
col1, col2 = st.columns(2)

# Plot Visualization 1 in the first column
with col1:
    plt.figure(figsize=(7, 6))
    plt.bar(['No', 'Yes'], average_rented_count.values, color=['orangered', 'chartreuse'])
    plt.title('Average Bicycle Rentals for Hit Sales', fontsize=14)
    plt.ylabel('Bicycle Rentals Count', fontsize=12) 
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    for i, v in enumerate(average_rented_count.values):
        plt.text(i, v, str(round(v)), fontsize=12, ha='center')
    st.pyplot(plt)

# Plot Visualization 2 in the second column
with col2:
    plt.figure(figsize=(3, 3))
    plt.pie(ratio_hit_sales, labels=['No Hit Sales', 'Hit Sales'], autopct='%1.1f%%', colors=['orangered', 'chartreuse'], explode=[0, 0.1])
    st.pyplot(plt)


# Visualization 3: Group by hour and calculate the average rented bike count
hourly_rentals = df.groupby('Hour')['Rented_Bike_Count'].mean()
plt.figure(figsize=(15, 6))
plt.plot(hourly_rentals.index, hourly_rentals.values, marker='o')
plt.title('Average Bicycle Rentals Throughout the Day', fontsize=16)
plt.xlabel('Time (24Hrs)', fontsize=12)
plt.ylabel('Number of Bicycle Rentals', fontsize=14)
plt.xticks(hourly_rentals.index, fontsize=14)
plt.yticks(fontsize=13)
plt.grid(True)
st.pyplot(plt)

# Visualization 4: Bicycle Rentals Across the Years
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
yearly_monthly_rentals = df.groupby(['Year', 'Month'])['Rented_Bike_Count'].sum()
plt.figure(figsize=(15, 7))
for year in yearly_monthly_rentals.index.get_level_values('Year').unique():
    monthly_rentals = yearly_monthly_rentals[year]
    if year == 2018:
        color = 'cyan'
    elif year == 2019:
        color = 'blue'
    plt.bar(month_order, monthly_rentals.reindex(month_order), label=str(year), color=color)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Number of Bicycle Rentals', fontsize=13)
plt.title('Bicycle Rentals Across the Years', fontsize=16)
plt.legend(title='Year')
plt.xticks(fontsize=14)
plt.yticks(fontsize=13)
plt.gca().yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')
st.pyplot(plt)

# Visualization 5: Bicycle Rentals Across Temperature Ranges
plt.figure(figsize=(14, 6))
df.Temperature.plot(kind='hist', weights=df['Rented_Bike_Count'], color='skyblue', bins=12, alpha=0.5, label='Temperature')
df.Dewpoint_Temp.plot(kind='hist', weights=df['Rented_Bike_Count'], color='green', bins=12, alpha=0.5, label='Dewpoint Temp')
plt.title('Bicycle Rentals Across Temperature Ranges', fontsize=16)
plt.xlabel('Temperature Range (Degree Celsius)', fontsize=13)
plt.ylabel('Number of Bicycle Rentals', fontsize=13)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=12)
plt.gca().yaxis.set_major_formatter('{:.0f}'.format)
plt.gca().yaxis.grid(True, linewidth=0.3, color='gray')
st.pyplot(plt)
