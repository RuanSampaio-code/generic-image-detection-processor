
#Impotando as bibliotecas necessárias
from ultralytics import YOLO
import cv2
import os

#Carregando o modelo pré-treinado YOLOv8n, é um modelo de detecção de objetos leve e rápido
modelo = YOLO("yolov8n.pt")
caminho = "assets/images/foto_teste_2.png"

#Definindo a função para processar a imagem
def processar_imagem(caminho):
        
    #Verificando se arquivo de imagem existe
    if not os.path.exists(caminho):
        print(f"Erro: Arquivo '{caminho}' não encontrado.")
    else:
        #Caso exista:
        #Ler imagem usando OpenCV
        imagem_cv2 = cv2.imread(caminho)

        #Realizar a detecção de objetos na imagem usando o modelo YOLOv8n. 
        resultados = modelo.predict(source=imagem_cv2, conf=0.5)

        #Percorre os resultados da detecção de objetos e exibe as informações relevantes.
        for r in resultados:

            img_plotada = r.plot()

            largura_desejada = 600
            proporcao = largura_desejada / img_plotada.shape[1]
            altura_desejada = int(img_plotada.shape[0] * proporcao)
            
            dimensoes = (largura_desejada, altura_desejada)
            img_menor = cv2.resize(img_plotada, dimensoes, interpolation=cv2.INTER_AREA)
            
            cv2.imshow("Process of image", img_menor)

            
            for box in r.boxes:
                #varaivies dentro de resultados
                classe = modelo.names[int(box.cls)]
                conf = float(box.conf)
                print(f"Detectado: {classe} | Confiança: {conf:.2f}")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

#Chamando a função para processar a imagem
processar_imagem(caminho)