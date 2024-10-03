import cv2
import face_recognition as fr

# cargar imagenes
foto_control = fr.load_image_file('yo1.jpg')
foto_prueba = fr.load_image_file('FotoA.jpg')

# pasar la forma en que las imagenes procesan el color: fotos con formato RGB
foto_control = cv2.cvtColor(foto_control, cv2.COLOR_BGR2RGB)
foto_prueba = cv2.cvtColor(foto_prueba, cv2.COLOR_BGR2RGB)

# Renococer en que parte de las fotos hay caras
lugar_cara1 = fr.face_locations(foto_control)[0]
lugar_cara2 = fr.face_locations(foto_prueba)[0]

# Codificar la cara que demos encontrado
cara_codificada1 = fr.face_encodings(foto_control)[0]
cara_codificada2 = fr.face_encodings(foto_prueba)[0]

# Mostrar rectangulos
cv2.rectangle(foto_control,
              (lugar_cara1[3], lugar_cara1[0]),
              (lugar_cara1[1], lugar_cara1[2]),
              (0, 255, 0),
              2)

cv2.rectangle(foto_prueba,
              (lugar_cara2[3], lugar_cara2[0]),
              (lugar_cara2[1], lugar_cara2[2]),
              (0, 255, 0),
              2)

# Comparar imagenes
resultado = fr.compare_faces([cara_codificada1], cara_codificada2)
print(resultado)

# medida de la distancia
distancia = fr.face_distance([cara_codificada1], cara_codificada2)
print(distancia)

# mostrar resultado de la distancia en la pantalla
cv2.putText(foto_prueba,
            f'{resultado} {distancia.round(2)}',
            (50, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0,250,0),
            2)

# mostrar imagenes
cv2.imshow('Foto Control', foto_control)
cv2.imshow('Foto Prueba', foto_prueba)

# Mantener el programa abierto
cv2.waitKey(0)


