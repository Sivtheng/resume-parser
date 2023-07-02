import joblib
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Get the absolute path of the script file
script_path = os.path.abspath(__file__)

# Get the directory of the script file
script_dir = os.path.dirname(script_path)

# Get the absolute path of the files
model_path = joblib.load(os.path.join(script_dir, '../models/trained_model.joblib'))
vectorizer_path = joblib.load(os.path.join(script_dir, '../models/vectorizer.joblib'))
label_encoder_path = joblib.load(os.path.join(script_dir, '../models/label_encoder.joblib'))


# Define the clean_text function (replace with your own cleaning/preprocessing function)
def clean_text(resume_text):
    # Implement your cleaning/preprocessing logic here
    cleaned_text = resume_text.lower()  # Example: Convert text to lowercase
    return cleaned_text

# Define the resume parsing function
def parse_resume(resume_text):
    # Preprocess the resume text
    cleaned_resume = clean_text(resume_text)
    
    # Vectorize the cleaned resume
    resume_vector = vectorizer_path.transform([cleaned_resume])
    
    # Make predictions
    predicted_category = model_path.predict(resume_vector)
    
    # Convert the predicted category code back to the original label
    predicted_category = label_encoder_path.inverse_transform(predicted_category)
    
    return predicted_category

# Example usage
if __name__ == '__main__':
    # Example resume text
    # Example resume text
    resume_text = "Jane Smith\n\nObjective: Experienced marketing professional with a strong background in digital marketing and social media management. Seeking a challenging role in a fast-paced marketing agency to utilize my skills in creating effective marketing campaigns and driving brand growth.\n\nEducation:\n- Bachelor of Business Administration, Marketing, XYZ University, 2016\n- Relevant coursework: Marketing Strategy, Consumer Behavior, Digital Marketing\n\nExperience:\n- Digital Marketing Specialist, ABC Company, 2017-2020\n- Developed and executed digital marketing campaigns across various platforms\n- Managed social media accounts and grew the follower base by 30%\n\nSkills:\n- Digital marketing strategies and tactics\n- Social media management and advertising\n- Google Analytics and SEO\n- Content creation and copywriting\n\nProjects:\n- Led a successful influencer marketing campaign resulting in a 20% increase in website traffic\n- Implemented email marketing automation, resulting in a 15% increase in email open rates\n\nReferences available upon request."

    # Parse the resume
    predicted_category = parse_resume(resume_text)
    
    # Print the predicted category
    print("Predicted Category:", predicted_category)
