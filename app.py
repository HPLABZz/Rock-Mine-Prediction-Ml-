from flask import Flask
from flask import render_template
from sonar import predict_result_KNN, predict_result_RF

app = Flask(__name__)

@app.route('/')
def KNN():
    result_KNN = predict_result_KNN()
    result_RF = predict_result_RF()

    return render_template('index.html',final_resultKNN=result_KNN, final_resultRF=result_RF)

if __name__ == "__main__":
    app.run(debug=True)
