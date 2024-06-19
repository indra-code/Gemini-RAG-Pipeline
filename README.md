Make sure that anaconda is installed on your machine.

Command for creating a virtual environment - conda create -p venv python == {your python version, prefarably > 3.10}

Command for activating virtual environment - conda activate venv/

Create a file named .env and paste your Gemini API Key as:

GOOGLE_API_KEY = {your api key}

Command for installing required dependencies on you virtual environment:

pip install -r requirements.txt

Command for running your streamlit application:

streamlit run app.py
