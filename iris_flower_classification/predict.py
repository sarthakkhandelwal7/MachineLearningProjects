import pickle
import numpy as np

model = pickle.load(open('iris_flower_classifier','rb'))
array = np.array([[5.1,3.5,1.4,0.2]])
result = model.predict(array)
print(result)