from flask import Flask
from flask import render_template, request
from PIL import Image
from io import BytesIO
import dog_app_functions
import os
import base64

app = Flask(__name__)

# Starting Page with picture upload option
@app.route("/")
def master():

    return render_template('master.html')


# Web page that displays classification results
@app.route('/go',methods=['POST'])
def go():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = Image.open(file.stream)

            # Convert the image to a byte array
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()

            # Convert the byte array to a base64 string, to display the image in html later
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    
    # Detect whether image shows a human or a dog and which breed resembles most
    breed, human_dog, result_string, breed_path = dog_app_functions.human_dog_classifier(img)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        breed=breed,
        human_dog=human_dog,
        result_string=result_string,
        breed_path= breed_path,
        img=img_base64
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()