Usage
Step 1: Start the Server
Run the checker_fast.py script to start the FastAPI server:

bash
Copy code
python checker_fast.py
This starts the server on http://localhost:8000. The server is now ready to accept requests.

Step 2: Test the API
Open Swagger UI: Open your browser and navigate to http://localhost:8000/docs. This page provides an interactive interface to test the API endpoints.

Locate the /compare-documents/ Endpoint: Scroll to the /compare-documents/ endpoint in the Swagger UI.

Try it Out:

Click the "Try it out" button.
You will see options to upload two files.
Upload Files:

requirements.xlsx: Contains the requirements to be validated.
responses.xlsx: Contains the responses to be compared.
Execute the API Call:

After uploading the files, click the "Execute" button.
The results of the comparison will appear in the response section.
Step 3: View the Results
The response will show:

Which requirements have been addressed.
Any gaps or mismatches between the requirements and the responses.
Sample Files
This repository includes the following sample files for testing:

requirements.xlsx: Contains example requirements.
responses.xlsx: Contains example responses for comparison.
