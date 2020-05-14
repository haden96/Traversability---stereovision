import numpy as np
import cv2
import funkcje
import os
import time
import dane

def kalibracja(liczba_zdjec):
    global szachownica_prawa, szachownica_lewa
    kryteria_zakonczenia_iteracji = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    pusta_tablica = np.zeros((9 * 6, 3), np.float32)
    pusta_tablica[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Tablice do przechowywania punktów
    punkty_3d = []
    punkty_prawa_kamera = []
    punkty_lewa_kamera = []
    print("Kalibracja rozpoczęta")

    for i in range(0, (liczba_zdjec - 1)):
        numer = str(i)
        # Wczytanie wczesniej zrobionych zdjec
        szachownica_prawa = cv2.imread('Kalibracja/prawa/chessboard-R' + numer + '.png', 0)
        szachownica_lewa = cv2.imread('Kalibracja/lewa//chessboard-L' + numer + '.png', 0)
        czy_znaleziono_prawa, rogi_prawa = cv2.findChessboardCorners(szachownica_prawa, (9, 6),
                                                                     None)  # znalezienie rogow szachownicy
        czy_znaleziono_lewa, rogi_lewa = cv2.findChessboardCorners(szachownica_lewa, (9, 6), None)
        if czy_znaleziono_prawa & czy_znaleziono_lewa:  # dodanie wspolrzednych do tablic
            cv2.cornerSubPix(szachownica_prawa, rogi_prawa, (11, 11), (-1, -1), kryteria_zakonczenia_iteracji)
            cv2.cornerSubPix(szachownica_lewa, rogi_lewa, (11, 11), (-1, -1), kryteria_zakonczenia_iteracji)
            punkty_3d.append(pusta_tablica)
            punkty_prawa_kamera.append(rogi_prawa)
            punkty_lewa_kamera.append(rogi_lewa)

    _, macierz_kamery_prawa, wspolczynniki_znieksztalcenia_prawa, wektor_rotacji_prawa, wektor_przesuniecia_prawa = \
        cv2.calibrateCamera(punkty_3d, punkty_prawa_kamera, szachownica_prawa.shape[::-1], None, None)

    _, macierz_kamery_lewa, wspolczynniki_znieksztalcenia_lewa, wektor_rotacji_lewa, wektor_przesuniecia_lewa = \
        cv2.calibrateCamera(punkty_3d, punkty_lewa_kamera, szachownica_lewa.shape[::-1], None, None)

    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    wynik, macierz_kamery_lewa, wspolczynniki_znieksztalcenia_lewa, \
    macierz_kamery_prawa, wspolczynniki_znieksztalcenia_lewa, \
    macierz_rotacji, macierz_przemieszczenia, macierz_esencjonalna, macierz_fundamentalna = \
        cv2.stereoCalibrate(
            punkty_3d,
            punkty_lewa_kamera,
            punkty_prawa_kamera,
            macierz_kamery_lewa,
            wspolczynniki_znieksztalcenia_lewa,
            macierz_kamery_prawa,
            wspolczynniki_znieksztalcenia_prawa,
            szachownica_prawa.shape[::-1],
            kryteria_zakonczenia_iteracji,
            flags)
    # rl,rr-macierze rotacji,pl,pr-macierze projekcji(3D na wyprostowany obraz,roi-wspolna czesc obu obrazow)

    rl, rr, pl, pr, q, czesc_wspolna_lewa, czesc_wspolna_prawa = cv2.stereoRectify(macierz_kamery_lewa,
                                                                                   wspolczynniki_znieksztalcenia_lewa,
                                                                                   macierz_kamery_prawa,
                                                                                   wspolczynniki_znieksztalcenia_prawa,
                                                                                   szachownica_prawa.shape[::-1],
                                                                                   macierz_rotacji,
                                                                                   macierz_przemieszczenia,
                                                                                   0, (0, 0))

    mapa_lewa = cv2.initUndistortRectifyMap(macierz_kamery_lewa,
                                            wspolczynniki_znieksztalcenia_lewa,
                                            rl,
                                            pl,
                                            szachownica_lewa.shape[::-1],
                                            cv2.CV_16SC2)

    mapa_prawa = cv2.initUndistortRectifyMap(macierz_kamery_prawa,
                                             wspolczynniki_znieksztalcenia_prawa,
                                             rr,
                                             pr,
                                             szachownica_prawa.shape[::-1],
                                             cv2.CV_16SC2)

    zapisz_npy(macierz_przemieszczenia, "macierz_przemieszczenia", "Kalibracja")
    zapisz_npy(macierz_kamery_lewa, "macierz_kamery_lewa", "Kalibracja")
    zapisz_mapy(mapa_lewa, "mapa_lewa", "Kalibracja")
    zapisz_mapy(mapa_prawa, "mapa_prawa", "Kalibracja")
    zapisz_npy(czesc_wspolna_lewa, 'roiL', "Kalibracja")
    zapisz_npy(czesc_wspolna_prawa, 'roiR', 'Kalibracja')


def zapisz_npy(data, nazwa, folder):
    np.save(os.path.join(folder, nazwa), data)


def zapisz_mapy(data, nazwa, folder):
    data1 = data[1]
    data0 = data[0]
    zapisz_npy(data1, nazwa + '1', folder)
    zapisz_npy(data0, nazwa + '0', folder)


pomiary=[]
mnoznik_odleglosc = 1
def pomiar_odleglosci(event, x, y, flags, params):
    global pomiary
    if event == cv2.EVENT_LBUTTONDBLCLK:
        macierz_kamery_lewa = params[0]
        macierz_przemieszczenia = params[1]
        dysp = params[2]
        b = abs(macierz_przemieszczenia[0]) * 0.025
        f = macierz_kamery_lewa[0][0]
        cx = macierz_kamery_lewa[0][2]
        cy = macierz_kamery_lewa[1][2]
        b = b[0]
        d = dysp[y, x]
        print(f,cx,b)
        z = mnoznik_odleglosc * (f * b / d)  # Z
        xx = mnoznik_odleglosc * ((x - cx) * b / d - (b / 2))  # X
        yy = mnoznik_odleglosc * (y - cy) * b / d  # Y
        pomiary.append([z, xx, yy])
        print('Z:', z, 'm  X:', xx, 'm  Y:', yy, 'm   x:', x, 'y:', y, 'dysp:', d)

kalibracja(6)
print("Kalibracja ukończona")
KameraP, KameraL = funkcje.wczytaj_zrodlo(1)
macierz_kamery_lewa, macierz_przemieszczenia, mapa_lewa, mapa_prawa = dane.wyniki_kalibracji()
cv2.namedWindow('Obraz dysparycji')
funkcje.trackbars('Obraz dysparycji')
while True:
    start = time.time()
    ObrazP, ObrazL = funkcje.pojedyncza_klatka(KameraP, KameraL)
    dysp = funkcje.tworzenie_mapy_dysparycji(ObrazL, ObrazP, mapa_lewa, mapa_prawa, 0, True, False)
    # dysp = cv2.applyColorMap(dysp, cv2.COLORMAP_JET)
    cv2.imshow('Obraz dysparycji2', dysp)

    cv2.setMouseCallback("Obraz dysparycji2", mnoznik_odleglosci, [macierz_kamery_lewa, macierz_przemieszczenia, dysp])
    stop = time.time()
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break
KameraP.release()
KameraL.release()
cv2.destroyAllWindows()
