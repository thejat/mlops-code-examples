"""
The MIT License (MIT)

Copyright (c) 2019 Avinash Sajjanshetty <hi@avi.im>

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



# PyTorch Flask API


Check the demo [here](https://pytorch-imagenet.herokuapp.com/).

If you'd like to check a super simple API server, then check [this repo](https://github.com/avinassh/pytorch-flask-api).


## Requirements

Runs with Python-3.7.3

Install them from `requirements.txt`:

    pip install -r requirements.txt


requirements.txt has:
    Flask==1.0.3
    https://download.pytorch.org/whl/cpu/torch-1.0.0-cp37-cp37m-linux_x86_64.whl
    torchvision==0.2.1
    numpy==1.16.4
    Pillow==7.1.0

## Local Deployment

Run the server:

    python app.py


## Heroku Deployment

[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy?template=https://github.com/avinassh/pytorch-flask-api-heroku)


## License

The mighty MIT license. Please check `LICENSE` for more details.



"""


import os
import json
from flask import Flask, render_template, request, redirect
import io
from PIL import Image
from torchvision import models
import torchvision.transforms as transforms


def get_model():
    model = models.densenet121(pretrained=True)
    model.eval()
    return model


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


# ImageNet classes are often of the form `can_opener` or `Egyptian_cat`
# will use this method to properly format it so that we get
# `Can Opener` or `Egyptian Cat`
def format_class_name(class_name):
    class_name = class_name.replace('_', ' ')
    class_name = class_name.title()
    return class_name

def get_prediction(image_bytes):
    try:
        tensor = transform_image(image_bytes=image_bytes)
        outputs = model.forward(tensor)
    except Exception:
        return 0, 'error'
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


model = get_model()
imagenet_class_index = json.load(open('../data/imagenet/imagenet_class_index.json'))

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        class_name = format_class_name(class_name)
        return render_template('result.html', class_id=class_id,
                               class_name=class_name)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
