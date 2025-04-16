
Flight Price Prediction Model
This project aims to predict flight prices using machine learning techniques. By analyzing various features from historical flight data, the model provides accurate price estimations, assisting travelers in making informed decisions.

📁 Project Structure
Data_Train.xlsx: Contains the training dataset with historical flight information.

Test_set.xlsx: Holds the test dataset for evaluating the model's performance.

flight_price.ipynb: Jupyter Notebook detailing the data preprocessing, exploratory data analysis (EDA), model training, and evaluation processes.

requirements.txt: Lists all the Python dependencies required to run the project.

🚀 Getting Started
To set up and run the project locally, follow these steps:

Clone the Repository:

bash
Copy
Edit
git clone https://github.com/Nabajit-Das/Flight-Predication-Model.git
cd Flight-Predication-Model
Create a Virtual Environment (optional but recommended):

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Launch Jupyter Notebook:

bash
Copy
Edit
jupyter notebook flight_price.ipynb
🧠 Model Overview
The model leverages regression algorithms to predict flight prices based on features such as:

Airline

Source and Destination cities

Departure and Arrival times

Duration

Number of stops

Additional information

The notebook provides a step-by-step walkthrough, including data cleaning, feature engineering, model selection, and performance evaluation.

📊 Results
After training, the model's performance is evaluated using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE). Visualizations are provided to compare actual vs. predicted prices.

🤝 Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to fork the repository and submit a pull request.

📄 License
This project is open-source and available under the MIT License.