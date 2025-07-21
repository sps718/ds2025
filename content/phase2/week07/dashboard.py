
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("Willoz: Apartment Price Estimator")

st.sidebar.header("Apartment Features")

def user_input_features():
    bathrooms = st.sidebar.slider("Bathrooms", 1.0, 5.0, 1.0, step=1.0)
    bedrooms = st.sidebar.slider("Bedrooms", 1.0, 5.0, 1.0, step=1.0)
    square_feet = st.sidebar.slider("Size (sq ft)", 100, 2000, 800)

    state = st.sidebar.selectbox("State", [
        "AK","AL","AR","AZ","CA","CO","CT","DC","DE","FL","GA","HI","IA","ID","IL","IN","KS","KY",
        "LA","MA","MD","ME","MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ","NM","NV","NY","OH",
        "OK","OR","PA","RI","SC","SD","TN","TX","UT","VA","VT","WA","WI","WV","WY"
    ])

    # Binary amenities
    amenities = [
        "Cats", "Dogs", "None Allowed", "AC", "Alarm", "Basketball", "Cable or Satellite", "Clubhouse",
        "Dishwasher", "Doorman", "Elevator", "Fireplace", "Garbage Disposal", "Gated", "Golf", "Gym",
        "Hot Tub", "Internet Access", "Luxury", "Not Listed", "Parking", "Patio/Deck", "Playground",
        "Pool", "Refrigerator", "Storage", "TV", "Tennis", "View", "Washer Dryer", "Wood Floors"
    ]

    amenity_values = {amenity: st.sidebar.checkbox(amenity) for amenity in amenities}

    # Create input data
    data = {
        "bathrooms": bathrooms,
        "bedrooms": bedrooms,
        "square_feet": square_feet,
    }

    # State one-hot encoding
    for s in ["AK","AL","AR","AZ","CA","CO","CT","DC","DE","FL","GA","HI","IA","ID","IL","IN","KS",
              "KY","LA","MA","MD","ME","MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ","NM","NV",
              "NY","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VA","VT","WA","WI","WV","WY"]:
        data[f"state_{s}"] = 1 if s == state else 0

    data.update({k: int(v) for k, v in amenity_values.items()})

    return pd.DataFrame([data])

input_df = user_input_features()

if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.subheader("Estimated Monthly Rent")
    st.write(f"💰 USD {prediction:,.2f}")

if st.checkbox("Show model input data"):
    st.write(input_df)
