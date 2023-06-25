from django.shortcuts import render
from .models import UploadedFile
import tensorflow as tf
import numpy as np
from PIL import Image


def index(request):
    return render(request, 'mainapp/index.html')



def upload_file(request):
    """
        param: upload files JPG
        return: text answer
    """
    if request.method == 'POST':
        print('Начало')
        uploaded_file = request.FILES['file']
        file_instance = UploadedFile(file=uploaded_file)
        file_instance.save()

        # Обработка файла и добавление результата в базу данных
        model = tf.keras.models.load_model('static/model/pneumonia_6_full_98%.h5')
        img = Image.open(file_instance.file).resize((128, 128)).convert("L")
        img_np = np.array(img).reshape(-1,128,128,1)
        img_np = img_np /255
        predictions = model.predict(img_np)
        result = round(predictions[0][0])
        if result == 1:
            answer = 'Пневмония. Необходимо обратиться к врачу'
        else:
            answer = 'Вы здоровы!'
        file_instance.result = answer
        file_instance.save()

        return render(request, 'mainapp/response.html', {'answer': answer})
    return render(request, 'mainapp/index.html')

