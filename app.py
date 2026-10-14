import pickle
from flask import Flask, render_template_string, request
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained Naive Bayes model
with open('naive_bayes.pkl', 'rb') as f:
  model = pickle.load(f)

# HTML template with modern Tailwind CSS layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Future Prediction | Naive Bayes Model</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 min-h-screen py-10 px-4">
    <div class="max-w-3xl mx-auto bg-white rounded-2xl shadow-xl overflow-hidden">
        <!-- Header -->
        <div class="bg-gradient-to-r from-indigo-600 to-purple-600 py-6 px-8 text-white">
            <h1 class="text-3xl font-extrabold tracking-tight">Employee Prediction Portal</h1>
            <p class="text-indigo-100 mt-1">Powered by Flask & Scikit-Learn Naive Bayes</p>
        </div>

        <!-- Form Content -->
        <div class="p-8">
            <form action="/predict" method="POST" class="space-y-6">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    
                    <!-- Education -->
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Education</label>
                        <select name="Education" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                            <option value="Bachelors">Bachelors</option>
                            <option value="Masters">Masters</option>
                            <option value="PHD">PHD</option>
                        </select>
                    </div>

                    <!-- Joining Year -->
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Joining Year</label>
                        <input type="number" name="JoiningYear" placeholder="e.g. 2017" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    </div>

                    <!-- City -->
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">City</label>
                        <select name="City" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                            <option value="Bangalore">Bangalore</option>
                            <option value="Pune">Pune</option>
                            <option value="New Delhi">New Delhi</option>
                        </select>
                    </div>

                    <!-- Payment Tier -->
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Payment Tier</label>
                        <select name="PaymentTier" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                            <option value="1">Tier 1</option>
                            <option value="2">Tier 2</option>
                            <option value="3">Tier 3</option>
                        </select>
                    </div>

                    <!-- Age -->
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Age</label>
                        <input type="number" name="Age" placeholder="e.g. 28" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    </div>

                    <!-- Gender -->
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Gender</label>
                        <select name="Gender" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                        </select>
                    </div>

                    <!-- Ever Benched -->
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Ever Benched</label>
                        <select name="EverBenched" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                            <option value="No">No</option>
                            <option value="Yes">Yes</option>
                        </select>
                    </div>

                    <!-- Experience In Current Domain -->
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Experience in Current Domain (Years)</label>
                        <input type="number" name="ExperienceInCurrentDomain" placeholder="e.g. 3" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    </div>

                </div>

                <!-- Submit Button -->
                <div>
                    <button type="submit" class="w-full bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-semibold py-3 px-6 rounded-lg shadow-md hover:from-indigo-700 hover:to-purple-700 transition duration-300">
                        Predict Outcome
                    </button>
                </div>
            </form>

            {% if prediction_text is not none %}
            <div class="mt-8 p-6 rounded-xl bg-gray-50 border border-gray-200 text-center shadow-inner">
                <h2 class="text-sm font-semibold text-gray-500 uppercase tracking-wider">Prediction Result</h2>
                <p class="text-2xl font-bold mt-2 {% if 'Leave' in prediction_text %}text-red-600{% else %}text-emerald-600{% endif %}">
                    {{ prediction_text }}
                </p>
            </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""


@app.route('/')
def home():
  return render_template_string(HTML_TEMPLATE, prediction_text=None)


@app.route('/predict', methods=['POST'])
def predict():
  try:
    # Extract values from form input
    education = request.form['Education']
    joining_year = int(request.form['JoiningYear'])
    city = request.form['City']
    payment_tier = int(request.form['PaymentTier'])
    age = int(request.form['Age'])
    gender = request.form['Gender']
    ever_benched = request.form['EverBenched']
    exp_domain = int(request.form['ExperienceInCurrentDomain'])

    # Format input into a DataFrame matching model feature order
    input_data = pd.DataFrame([[
        education,
        joining_year,
        city,
        payment_tier,
        age,
        gender,
        ever_benched,
        exp_domain,
    ]],
                              columns=[
                                  'Education',
                                  'JoiningYear',
                                  'City',
                                  'PaymentTier',
                                  'Age',
                                  'Gender',
                                  'EverBenched',
                                  'ExperienceInCurrentDomain',
                              ])

    # Predict using the loaded Naive Bayes model
    prediction = model.predict(input_data)[0]

    # Map output (adjust labels if your encoding differs: 1 = Leave, 0 = Stay)
    result_text = (
        'Employee Likely to Leave'
        if prediction == 1
        else 'Employee Likely to Stay'
    )

    return render_template_string(
        HTML_TEMPLATE, prediction_text=result_text
    )

  except Exception as e:
    return render_template_string(
        HTML_TEMPLATE, prediction_text=f'Error in prediction: {str(e)}'
    )


if __name__ == '__main__':
  app.run(debug=True)
