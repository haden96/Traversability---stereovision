import cv2
import numpy as np
from skimage import morphology
import rysowanie

""" Stworzenie mapy oblegania na podstawie obrazów dysparycji """
def tworzenie_mapy(macierz_kamery_lewa, macierz_przemieszczenia, dysp_przeszkody, dysp_przejezdne, res, roi):
    size_z, size_x = rozmiar_mapy(roi, res)

    przejezdne_chmura = punkty_3d(macierz_kamery_lewa, macierz_przemieszczenia, dysp_przejezdne)
    przeszkody_chmura = punkty_3d(macierz_kamery_lewa, macierz_przemieszczenia, dysp_przeszkody)

    przeszkody_chmura = filtrowanie(przeszkody_chmura, size_z, size_x, res)
    przejezdne_chmura = filtrowanie(przejezdne_chmura, size_z, size_x, res)

    mapa = mapa_zajecia(przejezdne_chmura, size_z, size_x)
    mapa = mapa_przeszkody(mapa, przeszkody_chmura)

    return mapa

""" Korekcja obrazu przy pomocy mapy uzyskanej podczas kalibracji """
def korekcja_obrazu(obraz, mapa):
    po_kalibracji = cv2.remap(obraz, mapa[0], mapa[1], interpolation=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_CONSTANT)

    return cv2.cvtColor(po_kalibracji, cv2.COLOR_BGR2GRAY)


def mapa_dysparycji_compute(stereo,lewa,prawa):
    return stereo.compute(lewa, prawa)


def tworzenie_mapy_dysparycji(obraz_lewa, obraz_prawa, mapa_lewa=None, mapa_prawa=None, algorytm=0, rysuj=False,
                              kalibracja=False):
    # Wybór algorytmu oszacowania dysparycji
    if kalibracja:
        if algorytm == 0:
            stereo = stereo_sgbm_trackbars()
        else:
            stereo = stereo_bm()
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
        wls_filter.setLambda(cv2.getTrackbarPos('Lambda', 'Obraz dysparycji'))
        wls_filter.setSigmaColor(cv2.getTrackbarPos('Sigma', 'Obraz dysparycji') / 100)
    else:
        if algorytm == 0:
            stereo = stereo_sgbm()
        else:
            stereo = stereo_bm()
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
        wls_filter.setLambda(30000)
        wls_filter.setSigmaColor(0.8)

    stereo_r = cv2.ximgproc.createRightMatcher(stereo)

    # Korekcja obrazów
    if mapa_lewa and mapa_prawa is not None:
        szara_lewa = korekcja_obrazu(obraz_lewa, mapa_lewa)
        szara_prawa = korekcja_obrazu(obraz_prawa, mapa_prawa)
    else:
        szara_lewa = cv2.cvtColor(obraz_lewa, cv2.COLOR_BGR2GRAY)
        szara_prawa = cv2.cvtColor(obraz_prawa, cv2.COLOR_BGR2GRAY)

    if rysuj:
        rysowanie.rektyfikacja(obraz_lewa, obraz_prawa, szara_lewa, szara_prawa)

    kernel = (9, 9)
    # Rozmycie wygładza utworzone obrazy dysparycji
    szara_lewa = cv2.GaussianBlur(szara_lewa, kernel, 0)
    szara_prawa = cv2.GaussianBlur(szara_prawa, kernel, 0)

    # Stworzenie map dla obydwu obrazów
    mapa_dysparycji_lewa = mapa_dysparycji_compute(stereo, szara_lewa, szara_prawa)
    mapa_dysparycji_prawa = mapa_dysparycji_compute(stereo_r, szara_prawa, szara_lewa)

    # Filtr WLS wygładza mapę dysparycji oraz wypełnia puste obszary
    mapa_dysparycji = wls_filter.filter(mapa_dysparycji_lewa, szara_lewa, None, mapa_dysparycji_prawa)
    mapa_dysparycji = cv2.normalize(src=mapa_dysparycji, dst=None, beta=0, alpha=255,
                                    norm_type=cv2.NORM_MINMAX)

    mapa_dysparycji = mapa_dysparycji[20:460, 130:620]

    return np.uint8(mapa_dysparycji)


def tworzenie_v_dysparycji(img):
    height = img.shape[0]
    max_disp = 255
    vhist_vis = np.zeros((height, max_disp), np.float)
    for i in range(height):
        vhist_vis[i, ...] = cv2.calcHist(images=[img[i, ...]], channels=[0], mask=None, histSize=[max_disp],
                                         ranges=[0, max_disp]).flatten() / float(height)

    vhist_vis = np.array(vhist_vis * 255, np.uint8)
    vblack_mask = vhist_vis < 5
    vhist_vis[vblack_mask] = 0
    return vhist_vis


def tworzenie_u_dysparycji(img):
    width = img.shape[1]
    max_disp = 255
    uhist_vis = np.zeros((max_disp, width), np.float)

    for i in range(width):
        uhist_vis[..., i] = cv2.calcHist(images=[img[..., i]], channels=[0], mask=None, histSize=[max_disp],
                                         ranges=[0, max_disp]).flatten() / float(width)

    uhist_vis = np.array(uhist_vis * 255, np.uint8)
    ublack_mask = uhist_vis < 1
    uhist_vis[ublack_mask] = 0
    return uhist_vis

""" Operacje morfologiczne """
def zamykanie(obraz, rozmiar):
    maska = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rozmiar[0], rozmiar[1]))
    return cv2.morphologyEx(obraz, cv2.MORPH_CLOSE, maska, iterations=1)


def otwieranie(obraz, rozmiar):
    maska = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rozmiar[0], rozmiar[1]))
    return cv2.morphologyEx(obraz, cv2.MORPH_OPEN, maska, iterations=1)

""" Stworzenie i zastosowanie masek na obrazie dysparycji  """
def maska_przeszkod(u_dysparycja_binarna, mapa_dysparycji):
    x, y = np.where(u_dysparycja_binarna)
    mapa_przeszkod = np.zeros_like(mapa_dysparycji, dtype=np.uint8)
    xs, ys = np.where(mapa_dysparycji[:, y] == x)
    mapa_przeszkod[xs, y[ys]] = 255
    return mapa_przeszkod


def maska_przejezdnosc(v_dysparycja_binarna, mapa_dysparycji, rho, theta, szerokosc_lini):
    wspolrzedne = np.argwhere(v_dysparycja_binarna)
    warunek = wspolrzedne[:, 1] * np.cos(theta) + wspolrzedne[:, 0] * np.sin(theta)
    x = np.where((warunek >= rho - szerokosc_lini / 2) & (warunek <= rho + szerokosc_lini / 2))
    poprawne = wspolrzedne[x]
    maska = np.zeros_like(mapa_dysparycji, dtype=np.uint8)
    xs, ys = np.where(mapa_dysparycji[poprawne[:, 0]] == poprawne[:, 1, None])
    maska[poprawne[xs, 0], ys] = 255
    return maska


def tworzenie_maski(mapa_przeszkod, mapa_dysparycji):
    mapa_bezprzeszkod = np.logical_not(mapa_przeszkod)
    niepoprawne = mapa_dysparycji.copy()
    niepoprawne[:] = np.nan
    maska_przeszkody = np.where(mapa_przeszkod, mapa_dysparycji, niepoprawne)
    maska_wolne = np.where(mapa_bezprzeszkod, mapa_dysparycji, niepoprawne)
    return maska_przeszkody, maska_wolne


""" Interpretacja wyników U-V dysparycji i reprezentacja ich na obrazie dysparycji"""
def przeszkody(mapa_dysparycji, u_dysparycja_prog, rysuj=None):
    u_dysparycja = tworzenie_u_dysparycji(mapa_dysparycji)

    u_dysparycja_binarna2 = np.uint8(np.where(u_dysparycja > u_dysparycja_prog, 1, 0))
    u_dysparycja_binarna = np.uint8(
        morphology.remove_small_objects(u_dysparycja_binarna2.astype(bool), min_size=30, connectivity=3))

    u_dysparycja_binarna = cv2.dilate(u_dysparycja_binarna, np.ones((7, 7), np.uint8), iterations=1)

    mapa_przeszkod = maska_przeszkod(u_dysparycja_binarna, mapa_dysparycji)

    mapa_przeszkod = zamykanie(mapa_przeszkod, [12, 12])
    maska_przeszkody, maska_wolne = tworzenie_maski(mapa_przeszkod, mapa_dysparycji)

    if rysuj:
        cv2.imshow('Mapa U-dysparycji', u_dysparycja_binarna * 255)
        rysowanie.wyniki(u_dysparycja_binarna2, 'U-dysparycja')

    return maska_przeszkody, maska_wolne, u_dysparycja_binarna


def przejezdnosc(bez_przeszkod_dysparycja, szerokosc_lini=20, rysuj=None):
    # Uzyskiwanie rozbieżności V
    v_dysparycja = tworzenie_v_dysparycji(bez_przeszkod_dysparycja)

    # Znajdowanie lini korealcji ziemi przy zastosowaniu transformacji Hough'a
    v_dysparycja_binarna = (v_dysparycja > 5).astype(np.uint8)

    # Szukanie lini podloza
    linie = np.squeeze(cv2.HoughLines(v_dysparycja_binarna.astype(np.uint8), 1, np.pi / 180, v_dysparycja.min()))
    if linie is not None:
        linia = linie[(linie[:, 1] > 1.5) * (linie[:, 1] < 3.0), :]
        if linia.size == 0:
            linia = linie[0]
        else:
            linia = linia[0]

        rho = linia[0]
        theta = linia[1]

        mapa_przejezdnosc = maska_przejezdnosc(v_dysparycja_binarna, bez_przeszkod_dysparycja, rho, theta,
                                               szerokosc_lini)
    if rysuj:
        rysowanie.v_dysparycja(v_dysparycja_binarna, 'linia korelacji podłoza', szerokosc_lini, theta, rho)

    return mapa_przejezdnosc, theta


mnoznik_odleglosc = 1.0

""" Stworzenie chmury puntków w współrzędnych rzeczywistych """
def punkty_3d(macierz_kamery_lewa, macierz_przemieszczenia, obraz_dysparycji):
    # Parametry kamery
    b = abs(macierz_przemieszczenia[0]) * 0.025
    f = macierz_kamery_lewa[0][0]
    cx = macierz_kamery_lewa[0][2]

    u, v = np.mgrid[0:obraz_dysparycji.shape[0], 0:obraz_dysparycji.shape[1]]
    z = mnoznik_odleglosc * np.where(obraz_dysparycji < 3, obraz_dysparycji, f * b[0] / obraz_dysparycji)
    x = (v - cx) * (z / f) - b / 2
    return np.array(np.dstack((x, z))).reshape(-1, 2)


def filtrowanie(punkty, size_z, size_x, res):
    punkty = (punkty / res).astype(np.int32)
    punkty[:, 0] = punkty[:, 0] + int(size_x / 2)
    punkty = punkty[(punkty >= [0, 0]).all(1) & (punkty < [size_x, size_z]).all(1)]
    return punkty


def rozmiar_mapy(roi, res):
    size_z = int(roi[1] / res)
    size_x = int(roi[0] / res)
    return size_x, size_z


def mapa_zajecia(punkty, size_z, size_x):
    x = punkty[:, 0]
    z = punkty[:, 1]
    mapa = np.ones([size_z, size_x], dtype=np.uint8) * 127
    mapa[-z, x] = 255
    return mapa


def mapa_przeszkody(mapa, punkty):
    x = punkty[:, 0]
    z = punkty[:, 1]
    mapa[-z, x] = 0
    return mapa


""" Odczyt obrazów """
def wczytaj_zrodlo(numer):
    if numer == 1:
        kamera_p = cv2.VideoCapture(2)
        kamera_l = cv2.VideoCapture(1)
    elif numer == 2:
        kamera_p = cv2.VideoCapture('prawa.avi')
        kamera_l = cv2.VideoCapture('lewa.avi')
    elif numer == 3:
        kamera_p = False
        kamera_l = False
    return kamera_p, kamera_l


def pojedyncza_klatka(kamera_p, kamera_l):
    if kamera_l and kamera_p:
        _, obraz_p = kamera_p.read()
        _, obraz_l = kamera_l.read()
    else:
        obraz_p = cv2.imread('Prawa.png')
        obraz_l = cv2.imread('Lewa.png')

    return obraz_p, obraz_l


def maski(maska, bdysp, pdysp):
    maska = zamykanie(maska, [20, 20])
    rezultat = np.uint8(morphology.remove_small_objects(maska.astype(bool)), min_size=70, connectivity=3)
    rezultat = zamykanie(rezultat, [20, 20])
    masked_img = bdysp * np.logical_not(rezultat)
    masked_img += pdysp
    masked_img2 = bdysp * rezultat
    return masked_img, masked_img2


""" ALGORYTMY OSZACOWANIA DYSPARYCJI """


def stereo_sgbm_trackbars():
    rozmiar_okna = cv2.getTrackbarPos('BlockSize', 'Obraz dysparycji')
    stereo = cv2.StereoSGBM_create(minDisparity=cv2.getTrackbarPos('minDisp', 'Obraz dysparycji'),
                                   numDisparities=16 + cv2.getTrackbarPos('numDisp', 'Obraz dysparycji') * 16,
                                   blockSize=rozmiar_okna,
                                   uniquenessRatio=cv2.getTrackbarPos('uniqueness', 'Obraz dysparycji'),
                                   speckleWindowSize=cv2.getTrackbarPos('speckWin', 'Obraz dysparycji'),
                                   speckleRange=cv2.getTrackbarPos('speckRan', 'Obraz dysparycji'),
                                   disp12MaxDiff=cv2.getTrackbarPos('maxDiff', 'Obraz dysparycji'),
                                   P1=8 * 3 * rozmiar_okna ** 2,
                                   P2=32 * 3 * rozmiar_okna ** 2,
                                   preFilterCap=cv2.getTrackbarPos('prefilter', 'Obraz dysparycji'),
                                   mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    return stereo


def stereo_sgbm():
    rozmiar_okna = 3
    stereo = cv2.StereoSGBM_create(minDisparity=25,
                                   numDisparities=192,
                                   blockSize=3,
                                   uniquenessRatio=10,
                                   speckleWindowSize=50,
                                   speckleRange=16,
                                   disp12MaxDiff=100,
                                   P1=8 * 3 * rozmiar_okna ** 2,
                                   P2=32 * 3 * rozmiar_okna ** 2,
                                   mode=cv2.StereoSGBM_MODE_SGBM_3WAY)
    return stereo


def stereo_bm():
    stereo = cv2.StereoBM_create(numDisparities=160, blockSize=9)
    return stereo


def trackbars(nazwa):
    cv2.namedWindow(nazwa)
    cv2.createTrackbar('BlockSize', nazwa, 5, 320, nothing)
    cv2.createTrackbar("numDisp", nazwa, 8, 15, nothing)
    cv2.createTrackbar("minDisp", nazwa, 6, 100, nothing)
    cv2.createTrackbar('uniqueness', nazwa, 10, 500, nothing)
    cv2.createTrackbar('speckWin', nazwa, 50, 300, nothing)
    cv2.createTrackbar('speckRan', nazwa, 2, 20, nothing)
    cv2.createTrackbar('maxDiff', nazwa, 1, 5000, nothing)
    cv2.createTrackbar('Lambda', nazwa, 30000, 1000000, nothing)
    cv2.createTrackbar('Sigma', nazwa, 80, 1000, nothing)
    cv2.createTrackbar('prefilter', nazwa, 10, 50, nothing)

def nothing(x):
    pass

def ray_tracking(u_dysparycja):
    u_dysparycja = np.flip(u_dysparycja, axis=0)
    return np.where(np.cumsum(np.not_equal(u_dysparycja, 0), axis=0), u_dysparycja, 255)

def histogram_czas(hist_czas, stop, start, start2):
    czas1 = start2 - start
    czas2 = stop - start2
    czas3 = stop - start
    hist_czas.append([czas1, czas2, czas3])
    return hist_czas