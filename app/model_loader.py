import os
import joblib
import pandas as pd

# Load your trained Random Forest model
base_dir = os.path.dirname(os.path.abspath(__file__))
print(f'base_dir is: ${base_dir}')
model_path = os.path.abspath(os.path.join(base_dir, '..', 'models', 'top2_SVM.pkl'))
print(model_path)
model = joblib.load(model_path)

# Define all features the model expects
expected_features = ['num_of_prev_attempts', 'studied_credits', 'sum_click', 'date_registration', 'gender_F', 'gender_M', 'region_East Anglian Region', 'region_East Midlands Region', 'region_Ireland', 'region_London Region', 'region_North Region', 'region_North Western Region', 'region_Scotland', 'region_South East Region', 'region_South Region', 'region_South West Region', 'region_Wales', 'region_West Midlands Region', 'region_Yorkshire Region', 'highest_education_A Level or Equivalent', 'highest_education_HE Qualification', 'highest_education_Lower Than A Level', 'highest_education_No Formal quals', 'highest_education_Post Graduate Qualification', 'imd_band_0-10%', 'imd_band_10-20', 'imd_band_20-30%', 'imd_band_30-40%', 'imd_band_40-50%', 'imd_band_50-60%', 'imd_band_60-70%', 'imd_band_70-80%', 'imd_band_80-90%', 'imd_band_90-100%', 'imd_band_Not_Provided', 'age_band_0-35', 'age_band_35-55', 'age_band_55<=', 'disability_N', 'disability_Y']

def load_model_and_predict(form_data):
    # Convert form input to float or int as needed

    input_dict = {}
    # Set default value in case the form data is missing
    input_dict['highest_education_Lower Than A Level'] = 1.0 if form_data.get('highest_education') == 'Lower Than A Level' else 0.0
    input_dict['gender_M'] = 1.0 if form_data.get('gender') == 'M' else 0.0
    input_dict['gender_F'] = 1.0 if form_data.get('gender') == 'F' else 0.0
    input_dict['studied_credits'] = float(form_data.get('studied_credits', 0))
    input_dict['sum_click'] = float(form_data.get('sum_click', 0))
    input_dict['date_registration'] = float(form_data.get('date_registration', 0))
    input_dict['num_of_prev_attempts'] = float(form_data.get('num_of_prev_attempts', 0))

    # Handle IMD
    selected_band = form_data.get("imd_band")
    for band in ['0-10%', '10-20', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']:
        input_dict[f'imd_band_{band}'] = 1.0 if selected_band == band else 0.0
    
    input_dict['imd_band_Not_Provided'] = 1.0 if selected_band == 'Not Provided' else 0.0
    
    # Convert to DataFrame and fill missing columns since the total will be 40 features
    input_df = pd.DataFrame([input_dict])
    print(f'input_df is ${input_df}')
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_features]
    prediction = model.predict(input_df)[0]
    print(f'Prediction is: {prediction}')
    return 'Will Not Drop Out' if prediction == 1 else 'Will Drop Out'
