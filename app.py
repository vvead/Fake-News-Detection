from flask import Flask, escape, request, render_template
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences



# Load model and tokenizer
tokenizer= pickle.load(open('tokenizer2.pkl', 'rb')) 
model = pickle.load(open('model2.pkl', 'rb'))
#w2v_model = pickle.load(open('w2v_model2.pkl', 'rb'))
max_length = 1000

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

    
@app.route("/prediction", methods=['GET','POST'])
def prediction():
    if request.method == "POST":
        news = [request.form['news']]
        news = tokenizer.texts_to_sequences(news)
        news = pad_sequences(news, maxlen=max_length)
        val_pkl = news
        predict ='FAKE' if ((model.predict(val_pkl)>=0.5).astype(int)).all() == 0 else 'REAL'

        print(predict)
        return render_template("prediction.html", prediction_text="News headline is -> {}".format(predict))
        
    else: 
        return render_template("prediction.html")
    

if __name__ == "__main__":
    app.run()