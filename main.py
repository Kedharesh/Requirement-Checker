import os
from google.oauth2.service_account import Credentials
from vertexai.preview.generative_models import GenerativeModel, Image
import fitz  # PyMuPDF
import streamlit as st
from google.cloud import aiplatform
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
from google.cloud import storage
import pandas as pd
import io

def extraction(file_path, file_type):
    if file_type == 'pdf':
        doc = fitz.open(file_path)
        page = doc.load_page(0)  # Load the first page
        pix = page.get_pixmap()
        img_data = pix.tobytes("png")

        # Load the image for Vertex AI
        vertex_image = Image.from_bytes(img_data)

        # Create the model and generate content
        generative_multimodal_model = GenerativeModel("gemini-1.0-pro-vision")

        # Generate content
        response = generative_multimodal_model.generate_content(["Extract the complete text data from this PDF page:", vertex_image])

        if response.text:
            print("Extracted text:")
            print(response.text)
        else:
            print("No text was extracted from the image.")
        return response.text
    elif file_type == 'excel':
        df = pd.read_excel(file_path)
        return df.to_string(index=False)

def summary(response_original, response_changed):
    generative_model = GenerativeModel("gemini-1.5-flash-001")
    # Prepare the prompt
    prompt = f"""
        I have two lists of data extracted from an image or Excel file. Please analyze them strictly based on the values provided:

        Company Requirements:
        {response_original}

        Invoice Data:
        {response_changed}

        Please provide a structured comparison focusing ONLY on:

        1. EXACT MATCHES: List items that are completely identical between both lists
        2. VALUE CHANGES: Where the same item exists but with different values
        Format: "Item: [Original Value] → [New Value]"
        3. NEW ADDITIONS: Items that appear only in the Invoice
        4. DELETIONS: Items that appear only in the Requirements

        Rules for comparison:
        - Compare only the actual values and numbers
        - Ignore formatting differences
        - Don't make assumptions about missing data
        - Don't infer relationships that aren't explicitly shown
        - If something is unclear, mark it as "Unable to determine" rather than making assumptions

        Please structure the output in clear sections with bullet points for easy reading.
    """
    # Generate the content
    generated_response = generative_model.generate_content(prompt)
    print(generated_response.text) 
    return generated_response.text  

if __name__ == "__main__":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "focus-ensign-437302-v1-a14f1f892574.json"
    project = "focus-ensign-437302-v1"
    location = "us-central1"
    bucket_name = "client-user-storage"
    aiplatform.init(project=project,location=location)
    
    # Fix the typo and use os.path.join for cross-platform compatibility
    original_path = os.path.join("pdf_classification", "docs_original")
    changed_path = os.path.join("pdf_classification", "docs_changed")

    # Ensure directories exist
    os.makedirs(original_path, exist_ok=True)
    os.makedirs(changed_path, exist_ok=True)

    st.set_page_config(page_title="Invoice Checker")
    st.header("Match companies requirements with Invoice")

    with st.sidebar:
        st.title("Menu:")
        st.subheader("Requirements")
        file_original = st.file_uploader("Upload Requirements (PDF or Excel)", type=["pdf", "xlsx"])
        
        st.subheader("Invoice")
        file_changed = st.file_uploader("Upload Invoice (PDF or Excel)", type=["pdf", "xlsx"])

        if st.button("Submit & Process"):
            if file_original and file_changed:
                with st.spinner("Processing..."):
                    try:
                        # Save original file
                        original_file_path = os.path.join(original_path, file_original.name)
                        with open(original_file_path, "wb") as f:
                            f.write(file_original.getbuffer())

                        # Save changed file
                        changed_file_path = os.path.join(changed_path, file_changed.name)
                        with open(changed_file_path, "wb") as f:
                            f.write(file_changed.getbuffer())

                        # Extract text from both files
                        original_file_type = 'pdf' if file_original.name.endswith('.pdf') else 'excel'
                        changed_file_type = 'pdf' if file_changed.name.endswith('.pdf') else 'excel'

                        original_text = extraction(original_file_path, original_file_type)
                        changed_text = extraction(changed_file_path, changed_file_type)

                        st.session_state['original_text'] = original_text
                        st.session_state['changed_text'] = changed_text

                        st.success("Files processed successfully!")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
            else:
                st.error("Please upload both original and changed files.")

    if st.button("Generate Summary"):
        if 'original_text' in st.session_state and 'changed_text' in st.session_state:
            output = summary(st.session_state['original_text'], st.session_state['changed_text'])
            st.write("Summary of differences:", output)
        else:
            st.warning("Please process the files first.")