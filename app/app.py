from flask import Flask, render_template, request
from model_loader import load_model_and_predict

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        prediction = load_model_and_predict(form_data)
        return render_template('index.html', prediction=prediction, form_data=form_data)
    return render_template('index.html', prediction=None, form_data=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
