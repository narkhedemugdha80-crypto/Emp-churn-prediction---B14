import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained model
with open('naive.pkl', 'rb') as f:
    model = pickle.load(f)

# HTML Template with an attractive, modern UI using Tailwind CSS
TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Prediction Portal</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 min-h-screen text-slate-100 flex items-center justify-center p-4">
    <div class="max-w-2xl w-full bg-slate-900/80 backdrop-blur-xl border border-slate-800 rounded-2xl shadow-2xl overflow-hidden p-8">
        
        <div class="text-center mb-8">
            <h1 class="text-3xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-cyan-400">
                Naive Bayes Prediction App
            </h1>
            <p class="text-slate-400 text-sm mt-2">Enter the employee parameters below to generate a prediction.</p>
        </div>

        {% if prediction is not none %}
        <div class="mb-6 p-4 rounded-xl bg-indigo-950/60 border border-indigo-500/30 text-center animate-fade-in">
            <p class="text-xs uppercase tracking-wider text-indigo-300 font-semibold">Prediction Result</p>
            <p class="text-2xl font-bold text-cyan-300 mt-1">
                {% if prediction == 1 %}
                    <span class="text-emerald-400">Class 1 (Positive / Yes)</span>
                {% else %}
                    <span class="text-rose-400">Class 0 (Negative / No)</span>
                {% endif %}
            </p>
            <p class="text-xs text-slate-400 mt-1">Raw Output: {{ prediction }}</p>
        </div>
        {% endif %}

        <form action="/predict" method="POST" class="grid grid-cols-1 md:grid-cols-2 gap-5">
            
            <!-- Education -->
            <div>
                <label class="block text-xs font-medium text-slate-300 uppercase mb-1">Education (Encoded)</label>
                <input type="number" step="any" name="Education" required 
                    value="{{ form_data.Education if form_data else '' }}"
                    class="w-full bg-slate-800/60 border border-slate-700 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500">
            </div>

            <!-- JoiningYear -->
            <div>
                <label class="block text-xs font-medium text-slate-300 uppercase mb-1">Joining Year</label>
                <input type="number" name="JoiningYear" required 
                    value="{{ form_data.JoiningYear if form_data else '' }}"
                    class="w-full bg-slate-800/60 border border-slate-700 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500">
            </div>

            <!-- City -->
            <div>
                <label class="block text-xs font-medium text-slate-300 uppercase mb-1">City (Encoded)</label>
                <input type="number" step="any" name="City" required 
                    value="{{ form_data.City if form_data else '' }}"
                    class="w-full bg-slate-800/60 border border-slate-700 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500">
            </div>

            <!-- PaymentTier -->
            <div>
                <label class="block text-xs font-medium text-slate-300 uppercase mb-1">Payment Tier</label>
                <input type="number" name="PaymentTier" required 
                    value="{{ form_data.PaymentTier if form_data else '' }}"
                    class="w-full bg-slate-800/60 border border-slate-700 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500">
            </div>

            <!-- Age -->
            <div>
                <label class="block text-xs font-medium text-slate-300 uppercase mb-1">Age</label>
                <input type="number" name="Age" required 
                    value="{{ form_data.Age if form_data else '' }}"
                    class="w-full bg-slate-800/60 border border-slate-700 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500">
            </div>

            <!-- Gender -->
            <div>
                <label class="block text-xs font-medium text-slate-300 uppercase mb-1">Gender (Encoded)</label>
                <input type="number" name="Gender" required 
                    value="{{ form_data.Gender if form_data else '' }}"
                    class="w-full bg-slate-800/60 border border-slate-700 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500">
            </div>

            <!-- EverBenched -->
            <div>
                <label class="block text-xs font-medium text-slate-300 uppercase mb-1">Ever Benched (Encoded)</label>
                <input type="number" name="EverBenched" required 
                    value="{{ form_data.EverBenched if form_data else '' }}"
                    class="w-full bg-slate-800/60 border border-slate-700 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500">
            </div>

            <!-- ExperienceInCurrentDomain -->
            <div>
                <label class="block text-xs font-medium text-slate-300 uppercase mb-1">Experience In Current Domain</label>
                <input type="number" name="ExperienceInCurrentDomain" required 
                    value="{{ form_data.ExperienceInCurrentDomain if form_data else '' }}"
                    class="w-full bg-slate-800/60 border border-slate-700 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500">
            </div>

            <div class="md:col-span-2 mt-4">
                <button type="submit" class="w-full bg-gradient-to-r from-indigo-500 to-cyan-500 hover:from-indigo-600 hover:to-cyan-600 text-white font-semibold py-3 rounded-lg shadow-lg transition duration-200 cursor-pointer">
                    Predict Now
                </button>
            </div>
        </form>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(TEMPLATE, prediction=None, form_data=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features in the precise order trained by the model[cite: 1]
        features = [
            float(request.form['Education']),
            float(request.form['JoiningYear']),
            float(request.form['City']),
            float(request.form['PaymentTier']),
            float(request.form['Age']),
            float(request.form['Gender']),
            float(request.form['EverBenched']),
            float(request.form['ExperienceInCurrentDomain'])
        ]
        
        # Reshape for sklearn prediction
        final_features = np.array([features])
        prediction = model.predict(final_features)[0]
        
        return render_template_string(TEMPLATE, prediction=int(prediction), form_data=request.form)
    except Exception as e:
        return render_template_string(TEMPLATE, prediction=f"Error: {str(e)}", form_data=request.form)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
