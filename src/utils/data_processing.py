import re

# Clean the resume data by removing irrelevant characters, symbols, and formatting.
def clean_resume(resume_text):
    # Remove non-alphanumeric characters
    resume_text = re.sub(r'[^a-zA-Z0-9\s]', '', resume_text)
    # Remove extra whitespaces
    resume_text = re.sub(r'\s+', ' ', resume_text)
    # Convert to lowercase
    resume_text = resume_text.lower()
    return resume_text

# Extract relevant information from the resumes, such as section headings (e.g., education, experience) and their corresponding content.
def extract_sections(resume_text):
    sections = {}
    # Define the section headings and their corresponding patterns
    section_patterns = {
        'education': ['education', 'academic qualifications'],
        'experience': ['experience', 'work experience'],
        'skills': ['skills'],
        # Add more sections as needed
    }

    # Extract the content for each section
    for section, patterns in section_patterns.items():
        section_content = ''
        for pattern in patterns:
            matches = re.findall(pattern, resume_text, re.IGNORECASE)
            if matches:
                # Combine multiple matches into a single section content
                section_content = ' '.join(matches)
                break
        sections[section] = section_content

    return sections

def process_resumes(resume_texts):
    cleaned_resumes = [clean_resume(resume) for resume in resume_texts]
    extracted_sections = [extract_sections(resume) for resume in cleaned_resumes]
    return extracted_sections
