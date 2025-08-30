
import onnx
import numpy as np
import sys

input_path = sys.argv[1]
output_path = input_path.replace(".onnx", "_cut.onnx")
input_names = ["images"]
output_names = ["/model.22/cv3.0/cv3.0.2/Conv_output_0",
                "/model.22/cv2.0/cv2.0.2/Conv_output_0",
                "/model.22/cv3.1/cv3.1.2/Conv_output_0",
                "/model.22/cv2.1/cv2.1.2/Conv_output_0",
                "/model.22/cv3.2/cv3.2.2/Conv_output_0",
                "/model.22/cv2.2/cv2.2.2/Conv_output_0"]
onnx.utils.extract_model(input_path, output_path, input_names, output_names)
print("Output model : ",output_path)
