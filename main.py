import cv2
import funkcje
import dane
import rysowanie
import time

if __name__ == '__main__':
    """ Ustawienie parametrów programu """
    opcje = {
        'kamera': 2,  # Wybór obrazu kamery(1) , wideo (2) , obraz (3)
        'prog': 15,  # Liczba pikseli do uznania za przeszkodę U-dysparycja
        'szerokosc_lini': 30,  # Szerokosc lini w pikselach V-dysparycja
        'rozdzielczosc': 0.012,  # Rozdzielczość siatki zajęcia [m]
        'region_zainteresowania': [5, 5],  # Region zainteresowania [z,x] w [m]
        'rozmiar_mapy': [600, 600],  # Rozmiar mapy w pikselach
        'algorytm_dysparycji': 0  # Algorytm oszacowania przejezdności : 0-SGBM 1-BM
    }

    """ Wczytanie macierzy i map uzyskanych podczas kalibracji"""
    macierz_kamery_lewa, macierz_przemieszczenia, mapa_lewa, mapa_prawa = dane.wyniki_kalibracji()

    kamera_p, kamera_l = funkcje.wczytaj_zrodlo(opcje['kamera'])
    while True:

        """ Otrzymanie pojedyńczych klatek z wybranego źródła oraz stworzenie na ich podstawie obrazu dysparycji """
        obraz_p, obraz_l = funkcje.pojedyncza_klatka(kamera_p, kamera_l)
        obraz_dysparycji = funkcje.tworzenie_mapy_dysparycji(obraz_l, obraz_p, mapa_lewa, mapa_prawa,
                                                             opcje['algorytm_dysparycji'], False)

        """ Uzyskanie przeszkód metodą U-dysparycji oraz usunięcie ich z obrazu dysparycji """
        pdysp, bdysp, u_dysp_bin = funkcje.przeszkody(obraz_dysparycji, opcje['prog'], False)

        """ Uzyskanie obszaru opdpowiadającemu podłożu """
        maska_przejezdnosc, theta = funkcje.przejezdnosc(bdysp, opcje['szerokosc_lini'], False)

        """ Filtrowanie masek"""
        maska_przeszkody, maska_bezprzeszkod = funkcje.maski(maska_przejezdnosc, bdysp, pdysp)

        """ Stworzenie mapy oblegania"""
        mapa_oblegania = funkcje.tworzenie_mapy(macierz_kamery_lewa, macierz_przemieszczenia, maska_przeszkody,
                                          maska_bezprzeszkod,
                                          opcje['rozdzielczosc'], opcje['region_zainteresowania'])

        """ Wyświetlenie wyników """
        cv2.imshow('Obraz dysparycji', obraz_dysparycji)
        mapa_oblegania = rysowanie.mapa(mapa_oblegania, opcje['rozmiar_mapy'], opcje['rozdzielczosc'],
                                            opcje['region_zainteresowania'], 0.5)

        rezultat = rysowanie.wysietlenie_wynikow(obraz_l, maska_bezprzeszkod)

        """ Zapisanie wyników i zakończenie pracy programu """
        if cv2.waitKey(1) & 0xFF == ord('z'):
            dane.zapisz_wyniki(rezultat=rezultat, obraz_lewa=obraz_l, mapa_dysparycji=obraz_dysparycji,
                                  mapa_oblegania=mapa_oblegania)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            # dane.zapisz_xlsx(pomiary, 'Pomiary', 'Wyniki')
            break

    kamera_p.release()
    kamera_l.release()
    cv2.destroyAllWindows()
