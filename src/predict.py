import joblib
import os

# Get the absolute path of the script file
script_path = os.path.abspath(__file__)

# Get the directory of the script file
script_dir = os.path.dirname(script_path)

# Get the absolute path of the files
model_path = joblib.load(os.path.join(script_dir, '../models/trained_model.joblib'))
vectorizer_path = joblib.load(os.path.join(script_dir, '../models/vectorizer.joblib'))
label_encoder_path = joblib.load(os.path.join(script_dir, '../models/label_encoder.joblib'))


def clean_text(resume_text):
    cleaned_text = resume_text.lower() 
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
    resume_text = "Emily Brown\n\nObjective: Passionate graphic designer with expertise in branding and visual communication. Seeking a creative position to apply my skills in designing impactful visual solutions and contributing to brand identity.\n\nEducation:\n- Bachelor of Fine Arts, Graphic Design, XYZ University, 2019\n- Relevant coursework: Typography, Branding, Illustration\n\nExperience:\n- Graphic Designer, ABC Design Studio, 2019-2021\n- Created visual designs for various marketing materials, including logos, brochures, and websites\n- Collaborated with clients to understand design requirements and deliver high-quality solutions\n\nSkills:\n- Proficient in Adobe Creative Suite (Photoshop, Illustrator, InDesign)\n- Strong understanding of design principles and typography\n- Ability to work on multiple projects and meet deadlines\n\nProjects:\n- Designed a new brand identity for a startup, resulting in increased brand recognition\n- Created a series of illustrations for a children's book, receiving positive feedback from readers\n\nReferences available upon request."

    # Parse the resume
    predicted_category = parse_resume(resume_text)
    
    # Print the predicted category
    print("Predicted Category:", predicted_category)
