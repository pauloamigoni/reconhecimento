import numpy as np
import cv2
from keras.preprocessing import image

#-----------------------------
#opencv initialization

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
#-----------------------------
#
# inicializacao de reconhecedor de expressao de face
from keras.models import model_from_json
model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('facial_expression_model_weights.h5') #Pesos de Carga

#-----------------------------

emotions = ('bravo', 'repugnancia', 'medo', 'feliz', 'triste', 'surpresa', 'neutra')

while(True):
	ret, img = cap.read()
	#img = cv2.imread('C:/Users/IS96273/Desktop/hababam.jpg')

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	#print(faces) #locations of detected faces

	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #desenhar retangulo na imagem principal
		
		detected_face = img[int(y):int(y+h), int(x):int(x+w)] # rosto detectado
		detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transformar em escala de cinza
		detected_face = cv2.resize(detected_face, (48, 48)) #redimensionar para 48x48
		
		img_pixels = image.img_to_array(detected_face)
		img_pixels = np.expand_dims(img_pixels, axis = 0)
		
		img_pixels /= 255 #os pixels estao na escala de [0, 255]. normalize todos os pixels na escala de [0, 1]
		
		predictions = model.predict(img_pixels) #armazenar probabilidades de 7 expressoes
		
		#encontrar array indexado max 0: irritado, 1: nojo, 2: medo, 3: feliz, 4: triste, 5: surpresa, 6: neutro
		max_index = np.argmax(predictions[0])
		
		emotion = emotions[max_index]
		
		#escreva o texto da emocao acima do retangulo
		cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
		
		#processo no final do rosto detectado
		#-------------------------



	cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
	cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
	cv2.imshow("window", img)

	#cv2.imshow('img',img)

	if cv2.waitKey(1) & 0xFF == ord('q'): #pressione q para sair
		break

#matar coisas cv abertas		
cap.release()
cv2.destroyAllWindows()