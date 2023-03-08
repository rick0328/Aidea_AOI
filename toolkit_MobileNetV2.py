# Import your library
import numpy as np 
import tensorflow as tf
from tensorflow import image
from tensorflow import keras


class Toolkit(object):
    def __init__(self):
        # No code is allowed in this method
        pass

    def load_model(self):
        # Implement here 
        model_path ='{}-best-model.h5'.format('MobileV2')
        self.model = keras.models.load_model(model_path)
    def perform_inference(self):
        # Implement here
        data_path="data_list.csv"
        mcrr=np.loadtxt(data_path,dtype="str",delimiter=',')
        data_path="/data"
        final=np.array(["ID","Label"])
        answer_sheet=[]
        for i in mcrr:
            file_path=data_path+"/"+ i
            image_value = tf.io.read_file(file_path)
            img = tf.image.decode_png(image_value,channels=3)
            img = tf.image.resize(img, (160, 160))
            img = np.expand_dims(img, axis=0)
            img_final = tf.image.convert_image_dtype(img, tf.float32)
            img=tf.keras.applications.mobilenet_v2.preprocess_input(img_final)
            y_test_predprob = self.model.predict(img)
            y_test_pred = y_test_predprob.argmax(-1)
            answer_sheet.append([i,y_test_pred[0]])
        for i in answer_sheet:
            final=np.row_stack((final,i))
        np.savetxt("/output/result.csv", final, delimiter=",",fmt='%s')
        
        
