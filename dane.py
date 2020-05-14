import os
import glob
import pandas as pd
import cv2
import numpy as np


def zapisz_xlsx(data, nazwa, folder):
    df = pd.DataFrame(data)
    writer = pd.ExcelWriter(folder + '/' + nazwa + '.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()


def wczytaj_xlsx(folder, nazwa):
    path_base = os.path.dirname(os.path.abspath(__file__))
    path_base = os.path.join(path_base, folder)
    file_name = os.path.basename(nazwa + '.xlsx')
    full_path = os.path.join(path_base, file_name)
    df = pd.read_excel(full_path, 'Sheet1')
    return df


def zapisz_wyniki(**kwargs):
    folder = glob.glob("Badania/*")
    path = "Badania/" + str(len(folder))
    try:
        os.mkdir(path)
    except OSError:
        print("Nie udało się zapisać")
    else:
        for name, image in kwargs.items():
            cv2.imwrite(path + '/ ' + name + '.png', image)
        print("Zapisano w:  ", path)



def zapisz_npy(data, nazwa,folder):
    np.save(os.path.join(folder,nazwa),data)


def wczytaj_npy(folder, nazwa):
    return np.load(folder + '/' + nazwa + '.npy', allow_pickle=True)


def wczytaj_mapy(folder, nazwa):
    data0 = wczytaj_npy(folder, nazwa + '0')
    data1 = wczytaj_npy(folder, nazwa + '1')
    return [data0, data1]


def wyniki_kalibracji():
    macierz_kamery_lewa = wczytaj_npy("Kalibracja", "macierz_kamery_lewa")
    macierz_przemieszczenia = wczytaj_npy("Kalibracja", "macierz_przemieszczenia")
    mapa_lewa = wczytaj_mapy("Kalibracja", "mapa_lewa")
    mapa_prawa = wczytaj_mapy("Kalibracja", "mapa_prawa")
    return macierz_kamery_lewa, macierz_przemieszczenia, mapa_lewa, mapa_prawa
