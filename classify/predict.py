from datetime import datetime
import logging
import os
from pathlib import Path

from urllib.request import urlopen
from PIL import Image
import tensorflow as tf
import numpy as np

scriptpath = os.path.abspath(__file__)
scriptdir = os.path.dirname(scriptpath)
filename = os.path.join(scriptdir, "model.pb")
labels_filename = os.path.join(scriptdir, "labels.txt")

output_layer = "loss:0"
input_node = "Placeholder:0"

graph_def = tf.compat.v1.GraphDef()
labels = []
network_input_size = 0


def _initialize():
    """Initializes the global variables."""
    global labels, network_input_size
    
    if not labels:
        with tf.io.gfile.GFile(filename, "rb") as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
        with open(labels_filename, "rt") as lf:
            labels = [line.strip() for line in lf.readlines()]
        
        with tf.compat.v1.Session() as sess:
            input_tensor_shape = sess.graph.get_tensor_by_name(
                "Placeholder:0"
            ).shape.as_list()
            network_input_size = input_tensor_shape[1]
            logging.info(f"network_input_size = {network_input_size}")


def _log_msg(msg: str) -> None:
    """
    Logs a message with the current timestamp.

    Parameters
    ----------
    msg : str
        The message to log.
    """
    logging.info("%s: %s", datetime.now(), msg)


def _extract_bilinear_pixel(img, x, y, ratio, x_origin, y_origin):
    """
    Extracts a pixel from the image using bilinear interpolation.

    Parameters
    ----------
    img : Any
        The image to extract the pixel from
    x : int
        The x coordinate of the pixel to extract
    y : int
        The y coordinate of the pixel to extract
    ratio : float
        The ratio of the width to the height of the extracted square
    x_origin : int
        The x coordinate of the origin of the extracted square
    y_origin : int
        The y coordinate of the origin of the extracted square.

    Returns
    -------
    numpy.ndarray
        The extracted pixel.
    """
    x_delta = (x + 0.5) * ratio - 0.5
    x0 = int(x_delta)
    x_delta -= x0
    x0 += x_origin
    
    if x0 < 0:
        x0, x1, x_delta = 0, 0, 0.0
    elif x0 >= img.shape[1] - 1:
        x0 = x1 = img.shape[1] - 1
        x_delta = 0.0
    else:
        x1 = x0 + 1
    
    y_delta = (y + 0.5) * ratio - 0.5
    y0 = int(y_delta)
    y_delta -= y0
    y0 += y_origin
    if y0 < 0:
        y0, y1, y_delta = 0, 0, 0.0
    elif y0 >= img.shape[0] - 1:
        y0 = y1 = img.shape[0] - 1
        y_delta = 0.0
    else:
        y1 = y0 + 1
    
    # Get pixels in four corners
    bl = img[y0, x0]
    br = img[y0, x1]
    tl = img[y1, x0]
    tr = img[y1, x1]
    
    # Calculate interpolation
    b = x_delta * br + (1.0 - x_delta) * bl
    t = x_delta * tr + (1.0 - x_delta) * tl
    pixel = y_delta * t + (1.0 - y_delta) * b
    return pixel.astype(np.uint8)


def _extract_and_resize(img, target_size):
    """
    Extracts the center square from the image and resizes it to target_size.

    Parameters
    ----------
    img : numpy.ndarray
        The image to extract the center square from.
    target_size : tuple
        The target size of the extracted square.

    Returns
    -------
    numpy.ndarray
        The extracted and resized square.
    """
    determinant = img.shape[1] * target_size[0] - img.shape[0] * target_size[1]
    
    if determinant < 0:
        ratio = float(img.shape[1]) / float(target_size[1])
        x_origin = 0
        y_origin = int(0.5 * (img.shape[0] - ratio * target_size[0]))
    elif determinant > 0:
        ratio = float(img.shape[0]) / float(target_size[0])
        x_origin = int(0.5 * (img.shape[1] - ratio * target_size[1]))
        y_origin = 0
    else:
        ratio = float(img.shape[0]) / float(target_size[0])
        x_origin = y_origin = 0
    
    resize_image = np.empty(
        (target_size[0], target_size[1], img.shape[2]), dtype=np.uint8
    )
    for y in range(target_size[0]):
        for x in range(target_size[1]):
            resize_image[y, x] = _extract_bilinear_pixel(
                img, x, y, ratio, x_origin, y_origin
            )
    return resize_image


def _extract_and_resize_to_256_square(image):
    """
    Extracts the center 256x256 square from the image and resizes it to 256x256.

    Parameters
    ----------
    image : numpy.ndarray
        The image to extract the center square from.

    Returns
    -------
    numpy.ndarray
        The extracted and resized square.
    """
    h, w = image.shape[:2]
    _log_msg(f"extract_and_resize_to_256_square: {w}x{h} resized to 256x256")
    return _extract_and_resize(image, (256, 256))


def _crop_center(img, cropx, cropy):
    """Crop the center of an image.

    Parameters
    ----------
    img : Any
        The image to crop
    cropx : int
        Width of crop
    cropy : int
        Height of crop

    Returns
    -------
    numpy.ndarray
        Cropped image
    """
    h, w = img.shape[:2]
    startx = max(0, w // 2 - (cropx // 2) - 1)
    starty = max(0, h // 2 - (cropy // 2) - 1)
    _log_msg(f"crop_center: {w}x{h} to {cropx}x{cropy}")
    return img[starty: starty + cropy, startx: startx + cropx]


def _resize_down_to_1600_max_dim(image):
    """Resizes an image such that the max dimension is 1600px.

    Parameters
    ----------
    image : numpy.ndarray
        The image to resize.

    Returns
    -------
    image : numpy.ndarray
        The resized image.
    """
    max_dim = 1600
    w, h = image.size
    if h < max_dim and w < max_dim:
        return image
    
    new_size = (max_dim * w // h, max_dim) if (h > w) else (max_dim, max_dim * h // w)
    _log_msg(f"resize: {w}x{h} to {new_size[0]}x{new_size[1]}")
    method = Image.BICUBIC
    
    if max(new_size) / max(image.size) >= 0.5:
        method = Image.BILINEAR
    
    return image.resize(new_size, method)


def _convert_to_nparray(image):
    """Converts the image to a numpy array.

    Parameters
    ----------
    image : PIL.Image
        The image to convert.

    Returns
    -------
    numpy.ndarray
        The converted image.
    """
    # RGB -> BGR
    _log_msg("Convert to numpy array")
    image = np.array(image)
    return image[:, :, (2, 1, 0)]


def _update_orientation(image):
    """Update orientation based on EXIF tags.

    Parameters
    ----------
    image : Any
        Input image data.

    Returns
    -------
    ndarray
        Input image data with orientation updated.
    """
    if hasattr(image, "_getexif"):
        exif = image._getexif()
        exif_orientation_tag = 0x0112
        if exif is not None and exif_orientation_tag in exif:
            orientation = exif.get(exif_orientation_tag, 1)
            _log_msg("Image has EXIF Orientation: " + str(orientation))
            # orientation is 1 based, shift to zero based and flip/transpose based on
            # 0-based values
            orientation -= 1
            if orientation >= 4:
                image = image.transpose(Image.TRANSPOSE)
            if orientation in [2, 3, 6, 7]:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            if orientation in [1, 2, 5, 6]:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


def _predict_image(image):
    """Process image and classify it as a dog or cat.

    Parameters
    ----------
    image : Any
        Input image data.

    Returns
    -------
    str {'dog', 'cat'}
        Classified image. Can be either 'dog', or 'cat'.
    """
    try:
        if image.mode != "RGB":
            _log_msg("Converting to RGB")
            image.convert("RGB")
        
        w, h = image.size
        _log_msg(f"Image size: {w}x{h}")
        
        # Update orientation based on EXIF tags
        image = _update_orientation(image)
        
        # If the image has either w or h greater than 1600 we resize it down respecting
        # aspect ratio such that the largest dimension is 1600
        image = _resize_down_to_1600_max_dim(image)
        
        image = _convert_to_nparray(image)
        
        # Crop the center square and resize that square down to 256x256
        resized_image = _extract_and_resize_to_256_square(image)
        
        # Crop the center for the specified network_input_Size
        cropped_image = _crop_center(
            resized_image, network_input_size, network_input_size
        )
        
        tf.compat.v1.reset_default_graph()
        tf.import_graph_def(graph_def, name="")
        
        with tf.compat.v1.Session() as sess:
            return _image_prediction(sess, cropped_image)
    
    except Exception as e:
        _log_msg(str(e))
        return f"Error: Could not preprocess image for prediction. Exception: {e}"


def _image_prediction(sess, cropped_image):
    """
    Run the image through the model and return the top-1 result.

    Parameters
    ----------
    sess : tf.Session
        Tensorflow session where the graph is loaded.
    cropped_image : numpy.ndarray
        The current session.

    Returns
    -------
    str {'dog', 'cat'}
        The predicted class. Possible values are: "cat" and "dog".
    """
    prob_tensor = sess.graph.get_tensor_by_name(output_layer)
    (predictions,) = sess.run(prob_tensor, {input_node: [cropped_image]})
    
    result = []
    highest_prediction = None
    for p, label in zip(predictions, labels):
        truncated_probability = np.float64(round(p, 8))
        if truncated_probability > 1e-8:
            prediction = {"tagName": label, "probability": truncated_probability}
            result.append(prediction)
            if (
                not highest_prediction
                or prediction["probability"] > highest_prediction["probability"]
            ):
                highest_prediction = prediction
    
    response = {
        "created": datetime.utcnow().isoformat(),
        "predictedTagName": highest_prediction["tagName"],
        "prediction": result,
    }
    
    _log_msg(f"Results: {response}")
    return response


def predict_image_from_url(image_url):
    """Predict the image at the specified URL.

    Parameters
    ----------
    image_url : str
        The URL of the image to be processed.

    Returns
    -------
    str {'dog', 'cat'}
        The prediction result. Possible values are: "cat" and "dog".
    """
    logging.info(f"Predicting from url: {image_url}")
    
    _initialize()
    
    if Path(image_url).is_file():
        image = Image.open(image_url)
        return _predict_image(image)
    
    with urlopen(image_url) as testImage:
        image = Image.open(testImage)
        return _predict_image(image)
