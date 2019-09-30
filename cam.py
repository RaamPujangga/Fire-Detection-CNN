import cv2
import numpy as np
import os, sys
import tensorflow as tf
import webbrowser



 
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(0)
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()): 
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
        cv2.imwrite('frame.jpg',frame)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# change this as you see fit
        image_path = "C:/tensorflow1/frame.jpg"

# Read in the image_data
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        
# Loads label file, strips off carriage return
        label_lines = [line.rstrip() for line 
                           in tf.gfile.GFile("C:/tmp/output_labels.txt")]

# Unpersists graph from file
        with tf.gfile.FastGFile("C:/tmp/output_graph.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

        with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
            predictions = sess.run(softmax_tensor, \
                      {'DecodeJpeg/contents:0': image_data})
    
    # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                print('%s (score = %.5f)' % (human_string, score))
                if human_string == ('fire') and score >= (0.8):
                  webbrowser.open('https://api.telegram.org/bot944415410:AAFCIgbTCjs-_ZAEPz4YciCGzg5mX_FFF9M/sendMessage?chat_id=389776309&text=Your room detected a fire, Caution Wildfire!!')
                
 
    # Display the resulting frame
        
        
    # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()
