import requests

#Post response from the server
sample_dict = {     'workclass': 'state_gov',
                    'education': 'bachelors',
                    'marital_status': 'never_married',
                    'occupation': 'adm_clerical',
                    'relationship': 'not_in_family',
                    'race': 'white',
                    'sex': 'male',
                    'native_country': 'united_states',
                    'age': 39,
                    'fnlwgt': 77516,
                    'education_num': 13,
                    'capital_gain': 2174,
                    'capital_loss': 0,
                    'hours_per_week': 40
                }
url = "https://udacity-c3-project.herokuapp.com/predict"
post_response = requests.post(url, json=sample_dict)
print(post_response.status_code)
print(post_response.content)
