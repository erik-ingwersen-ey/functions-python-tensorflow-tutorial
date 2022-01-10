# functions-python-tensorflow-tutorial

Demo of azure functions. This demo, shows how to set up a simple image 
classification model using TensorFlow and Azure Functions. It classifies cats and 
dogs images.

**Original tutorial:** [functions-python-tensorflow-tutorial](https://github.com/Azure-Samples/functions-python-tensorflow-tutorial)

All credits to the original authors, and the Azure team.

**See it in action, predicting my own pet (Bubba 🐶):**

![](./resources/assets/usage-demo.gif)

## Modifications

From the original tutorial, the following modifications were made:

* Updated the required TensorFlow version from 1.14.0 to the latest version, and 
  updated the imports to reflect the new version.
* Broken down the original functions inside `predict.py` to make them more readable.
* Added docstrings to every function inside `predict.py`.
* Modified `__init__.py` from `classify`, to allow using local images, as well.

## Setup

Setting up this demo is quite simple. You need to have a Python environment 
installed on your machine. You can install it with the following command:

```console
$ python -m venv venv
```

Then, you can activate the environment with:

```console
$ source venv/bin/activate
```

Finally, you can install the required packages with:

```console
$ pip install -r requirements.txt
```

And you are ready to go!

## Running the demo

Now that you have every thing up and running, you can run the demo with:

```console
$ func start --verbose 
```

> Note that the `--verbose` flag is optional. It prints more stuff that helps debug
> to the console.

At another console window, you can run the following command to start the frontend:

```console
$ python -m http.server
```

And you are ready to go! Now open your browser and go to this [link](http://localhost:8000/frontend/)
to start making predictions.

Want an example to give to the model? Try asking it to predict [this picture](https://github.com/ingwersen-erik/PREFS/blob/main/img/IMG_0469.png?raw=true) of my dog Bubba 🐶.


## Notes

### Classify Function

The whole model lives inside the `classify` folder. As recommended by Microsoft, 
every Azure function you create should have its own directory.

### Model

The trained model used in this demo is a pre-trained TensorFlow model, found at `classify/model.pb`. 

### Function Overview

Aside from the pre-trained model at `classify/model.pb`, the real brains of the
operation live inside the `predict.py` file. There, images you upload to the model 
are processed and normalized, according to model requirements. Then, still inside 
the `predict.py` file, the model is loaded, the picture uploaded, and finally the 
prediction is made.

If you want to know more about each function, you can read the docstring of each 
function.

Last but not least, the `__init__.py` file contains the code that azure functions 
needs and is looking for. Each azure function must have a `__init__.py` file (or similar),
and a `functions.json` file. The last one is the configuration file for the function, 
which is where you can specify the name of the function, the code to run, the inputs 
required, and the outputs that the function will return.

A little hint when designing your functions: design them to be [idempotent](https://www.google.com/url?
sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiO2dWW5af1AhXur5UCHSOTAYgQmhN6BAgrEAI&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FIdempotence&usg=AOvVaw2tBKzMM7JWe5m8N5lGXEiY) if you 
want them to work every time, and have a long life.


## Advanced usage

### Train Custom Model

If you want to fly solo, and train your own model, follow the [tutorial](./train-custom-vision-model.md)

## Credits

Full credits go to the original authors, and the Azure team. These are the authors 
I was able to find in the original tutorial (sorry if I missed someone):

* [anthonychu](https://github.com/anthonychu)
* [asavaritayal](https://github.com/asavaritayal)
* [craigshoemaker](https://github.com/craigshoemaker)
* [jlooper](https://github.com/jlooper)
* [v-rajagt](https://github.com/v-rajagt)


## FAQ

I've written this FAQ before any questions were asked. This means I don't know the 
efficacy of the questions. Nevertheless, here are some questions you might have, and 
the answers:

**Q:** What is a pre-trained model?
**A:** A model that has been previously trained on a dataset (training data), and is ready to make predictions.

**Q:** What if I upload a picture that contains neither a dog nor a cat?
**A:** There's a saying in data science: trash-in = trash-out. If you give the 
model an example of something it wasn't trained to predict, well it'll give you a trash answer. Just remember: this isn't some super-ai model with good-like knowledge.

**Q:** What if I upload a picture that contains a dog and a cat?
**A:** Well, in pre-processing, the center of the image is cropped out, so my guess is that if the dog is in the center, it'll predict dog, otherwise cat. But who knows. Try it out, and them tell me what happens.

**Q:** Using this model will cost me money?
**A:** No, as long as you use the model locally, your wallet should be fine. The setup shows how to run it locally, and I didn't even mentioned publishing the function to azure, to avoid you from 
unknowingly spending money. But Azure functions is pretty cool, and it provides a lot for your buck.

**Q:** Can I use this model for my own project?
**A:** Sure! Just credit the original authors, like I did. 
