import tkinter as tk
from PIL import Image, ImageTk
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv('dados_clima_treino.csv')
x = data[['temperatura', 'umidade', 'pressao']].values
y = data['chance de chuva'].values

model = Sequential()
model.add(Dense(10, input_dim=3, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=500, batch_size=10)

def obter_previsoes():
    temperatura_str = temperatura_entry.get().replace(',', '.')
    umidade_str = umidade_entry.get().replace(',', '.')
    pressao_str = pressao_entry.get().replace(',', '.')

    try:
        temperatura = float(temperatura_str)
        umidade = float(umidade_str)
        pressao = float(pressao_str)

        dados = [temperatura, umidade, pressao]
        novos_dados = [dados]
        previsoes = model.predict(novos_dados)
        chance_chuva = previsoes[0][0] * 100

        resultado_label.config(text=f"Chance de chuva: {chance_chuva:.2f}%")
        
        if (chance_chuva > 70):
            imagem_chuva = Image.open("chuva.png")
            imagem_chuva = imagem_chuva.resize((200, 200))  # Redimensiona a imagem

            imagem_chuva = ImageTk.PhotoImage(imagem_chuva)
            imagem_label.config(image=imagem_chuva)
            imagem_label.image = imagem_chuva
        else:
            imagem_label.config(image=None)
            imagem_label.image = None
    except ValueError:
        resultado_label.config(text="Valores inválidos")

# Cria a janela principal
window = tk.Tk()
window.title("Previsão de Chuva")
window.geometry("500x400")

temperatura_label = tk.Label(window, text="Temperatura:")
temperatura_label.pack()
temperatura_entry = tk.Entry(window)
temperatura_entry.pack()

umidade_label = tk.Label(window, text="Umidade:")
umidade_label.pack()
umidade_entry = tk.Entry(window)
umidade_entry.pack()

pressao_label = tk.Label(window, text="Pressão:")
pressao_label.pack()
pressao_entry = tk.Entry(window)
pressao_entry.pack()

resultado_label = tk.Label(window, text="")
resultado_label.pack()

imagem_label = tk.Label(window)
imagem_label.pack()

previsao_button = tk.Button(window, text="Obter Previsões", command=obter_previsoes)
previsao_button.pack()

window.mainloop()
