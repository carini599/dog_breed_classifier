from flask import Flask
from flask import render_template, request, jsonify

from PIL import Image
from io import BytesIO
from werkzeug.utils import secure_filename
import dog_app_functions
import os

app = Flask(__name__)


# index webpage displays cool visuals and receives user input text for model
#@app.route('/')
#@app.route('/master')
#def master():
#    return render_template(
#        'master.html',
#    )


@app.route("/")
def master():

    if os.path.exists("static/cache.jpeg"):
        print('Cache wird geleert')
        os.remove("static/cache.jpeg")
    return render_template('master.html')


# web page that handles user query and displays model results
@app.route('/go',methods=['POST'])
def go():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            print(filename)
            img = Image.open(file.stream)
            with BytesIO():
                img.save("static/cache.jpeg", "jpeg")

    
    
    # save user input in query
    # query = request.args.get('query', '') 

    # use model to predict classification for query
    #classification_results = query
    #print(classification_results)

    #Initialize Model 
    Xception_Model = dog_app_functions.initialize_model()
    breed, human_dog, result_string, breed_path = dog_app_functions.human_dog_classifier("static/cache.jpeg", Xception_Model)


    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        breed=breed,
        human_dog=human_dog,
        result_string=result_string,
        breed_path= breed_path
        #query=query,
        #classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()