import numpy as np
import cv2 as cv
from openvino.runtime import Core
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description='Run single image super resolution with OpenVINO')
parser.add_argument('-i', dest='input', help='Path to input image', required=True)
parser.add_argument('-m', dest='model', default='single-image-super-resolution-1033.xml', help='Path to the model XML file')
parser.add_argument('-o', dest='output', default='output.png', help='Path to save the output image')
args = parser.parse_args()

# Initialize OpenVINO runtime core
core = Core()
model = core.read_model(args.model)
compiled_model = core.compile_model(model, 'CPU')

# Get correct input tensor names and expected shape
input_tensor_0 = compiled_model.input(0)
input_tensor_1 = compiled_model.input(1)
expected_shape = input_tensor_0.shape  # Example: [1, 3, 360, 640]

# Read and resize image to match model's input size
img = cv.imread(args.input)
if img is None:
    raise ValueError(f"Cannot read the image at {args.input}")

inp_h, inp_w = expected_shape[2], expected_shape[3]  # Use model's expected size
out_h, out_w = inp_h * 3, inp_w * 3  # Keep this scaling factor

img_resized = cv.resize(img, (inp_w, inp_h))  # Resize to model input size

# Prepare input
inp = img_resized.transpose(2, 0, 1)  # Convert HWC to CHW
inp = inp.reshape(1, 3, inp_h, inp_w).astype(np.float32)

# Prepare second input - bicubic resize of first input
resized_img = cv.resize(img_resized, (out_w, out_h), interpolation=cv.INTER_CUBIC)
resized = resized_img.transpose(2, 0, 1)
resized = resized.reshape(1, 3, out_h, out_w).astype(np.float32)

# Infer the model
inputs = {input_tensor_0.any_name: inp, input_tensor_1.any_name: resized}
request = compiled_model.create_infer_request()
request.infer(inputs)
out = request.get_output_tensor(0).data  # Get the output tensor

# Process output
out = out.reshape(3, out_h, out_w).transpose(1, 2, 0)
out = np.clip(out * 255, 0, 255).astype(np.uint8)

# Save output image
cv.imwrite(args.output, out)
print(f"Output image saved to {args.output}")
