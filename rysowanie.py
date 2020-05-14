import cv2
import numpy as np
from matplotlib import pyplot as plt


def wyniki(obraz, tytul):
    plt.figure()
    plt.imshow(obraz, cmap='gray', vmin=0, vmax=255)
    plt.title(tytul)
    plt.savefig('Wyniki/' + tytul + '.png')
    plt.show()
    return 0


def region_zaintereowania(img, roi):
    ptk1 = roi[0:2]
    ptk2 = roi[2:4]
    color = (0, 255, 0)
    img = cv2.rectangle(img, (ptk2[0], ptk2[1]), (ptk1[0], ptk1[1]), color, 2)
    return img


def linie(img, skok):
    color = (0, 0, 255)
    wysokosc = img.shape[0]
    szerokosc = img.shape[1]
    for i in range(skok, wysokosc, skok):
        cv2.line(img, (0, i), (szerokosc, i), color, 1)
    return img


def rektyfikacja(img1, img2, img1_r, img2_r):
    img_con1 = np.concatenate((img1, img2), axis=1)
    img_con2 = np.concatenate((img1_r, img2_r), axis=1)
    wynik = np.concatenate((img_con1, img_con2), axis=0)
    img = linie(wynik, 40)
    small = cv2.resize(img, (0, 0), fx=0.85, fy=0.85)
    cv2.imshow('Rektyfikacja', small)


def kalibracja(img1, img2, nazwa):
    wynik = np.concatenate((img1, img2), axis=1)
    small = cv2.resize(wynik, (0, 0), fx=0.8, fy=0.8)
    cv2.imshow(nazwa, small)


def mapa(img, rozmiar, roz, roi, skok=40):
    x, z = img.shape
    fx = rozmiar[0] / x
    fy = rozmiar[1] / z
    nowa = cv2.resize(img, (0, 0), fx=fx, fy=fy)
    nowa = mapa_oblegania(nowa, skok, roz, roi)
    cv2.imshow('Mapa oblegania', nowa)
    return nowa


def mapa_oblegania(img, skok, roz, roi):
    color = (0, 0, 255)
    wysokosc = img.shape[0]
    szerokosc = img.shape[1]
    size_z = int(roi[1] / roz)
    test = roz / (wysokosc / size_z)
    podzialka = int(skok / test)
    xxx = roi[1]
    for i in range(podzialka, wysokosc, podzialka):
        cv2.line(img, (0, i), (szerokosc, i), color, 1)
        xxx -= skok
        img = cv2.putText(img, str(xxx), (10, i - 3), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def v_dysparycja(obraz, tytul, szerokosc_lini, theta_1, rho_1):
    plt.figure()
    plt.imshow(obraz)
    plt.title(tytul)
    x = np.arange(0., obraz.shape[1], 1)
    plt.plot(x, ((-np.cos(theta_1) / np.sin(theta_1)) * x + (rho_1 + szerokosc_lini / 2) / np.sin(theta_1)), 'b')
    plt.plot(x, ((-np.cos(theta_1) / np.sin(theta_1)) * x + (rho_1 - szerokosc_lini / 2) / np.sin(theta_1)), 'b')
    plt.plot(x, ((-np.cos(theta_1) / np.sin(theta_1)) * x + (rho_1 / np.sin(theta_1))), 'r--')
    plt.savefig('Wyniki' + tytul + '.png')
    plt.show()
    return 0


def wysietlenie_wynikow(obraz_lewa, maska_przejezdnosc):
    maska_przejezdnosc = maska_przejezdnosc.astype(np.bool) * 255
    maska = cv2.cvtColor(maska_przejezdnosc.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    maska[:, :, [0, 2]] = 0
    temp = np.zeros_like(obraz_lewa)
    temp[20:460, 130:620] = maska
    img1 = cv2.add(obraz_lewa, temp)
    cv2.imshow('Wynik', img1)
    return img1