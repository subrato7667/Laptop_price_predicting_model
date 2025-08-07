from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load pipeline and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pd.DataFrame(pickle.load(open("df.pkl", 'rb')))

@app.route('/')
def index():
    data_dict = {
        'company': sorted(df['Company'].unique()),
        'typename': sorted(df['TypeName'].unique()),
        'ram': sorted(df['Ram'].unique()),
        'weight': [round(df['Weight'].min(), 2), round(df['Weight'].max(), 2)],
        'os': sorted(df['os'].unique()),
        'touchscreen': sorted(df['Touchscreen'].unique(), reverse=True),
        'ips': sorted(df['Ips'].unique(), reverse=True),
        'cpu': sorted(df['Cpu brand'].unique()),
        'ssd': sorted(df['SSD'].unique()),
        'hdd': sorted(df['HDD'].unique()),
        'gpu': sorted(df['Gpu brand'].unique()),
        'resolution': [
            '1920x1080', '1366x768', '1600x900',
            '3840x2160', '3200x1800', '2880x1800',
            '2560x1600', '2560x1440', '2304x1440'
        ],
        'prediction': False
    }
    return render_template('index.html', data_dict=data_dict)


@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    company = request.form['company']
    typename = request.form['typename']
    ram = int(request.form['ram'])
    weight = float(request.form['weight'])
    touchscreen = int(request.form['touchscreen'])
    ips = int(request.form['ips'])
    screen_size = float(request.form['screen_size'])
    resolution = request.form['resolution']
    cpu = request.form['cpu']
    ssd = int(request.form['ssd'])
    hdd = int(request.form['hdd'])
    gpu = request.form['gpu']
    os_ = request.form['os']

    # Calculate ppi
    try:
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res**2 + Y_res**2) ** 0.5) / screen_size
    except:
        return "Invalid screen resolution or size"

    # Prepare input
    input_data = pd.DataFrame([[company, typename, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os_]],
        columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips', 'ppi',
                 'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os']
    )

    # Prediction
    try:
        predicted_price = np.exp(pipe.predict(input_data)[0])  # Reverse log
        formatted_price = f"₹ {predicted_price:,.2f}"
    except Exception as e:
        return f"Prediction Error: {str(e)}"

    # Return updated data dict
    data_dict = {
        'company': sorted(df['Company'].unique()),
        'typename': sorted(df['TypeName'].unique()),
        'ram': sorted(df['Ram'].unique()),
        'weight': [round(df['Weight'].min(), 2), round(df['Weight'].max(), 2)],
        'os': sorted(df['os'].unique()),
        'touchscreen': sorted(df['Touchscreen'].unique(), reverse=True),
        'ips': sorted(df['Ips'].unique(), reverse=True),
        'cpu': sorted(df['Cpu brand'].unique()),
        'ssd': sorted(df['SSD'].unique()),
        'hdd': sorted(df['HDD'].unique()),
        'gpu': sorted(df['Gpu brand'].unique()),
        'resolution': [
            '1920x1080', '1366x768', '1600x900',
            '3840x2160', '3200x1800', '2880x1800',
            '2560x1600', '2560x1440', '2304x1440'
        ],
        'prediction': formatted_price
    }

    return render_template('index.html', data_dict=data_dict)

if __name__ == '__main__':
    app.run(debug=True)
