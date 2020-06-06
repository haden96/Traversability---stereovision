import cv2
import funkcje
i = 0

kryteria_zakonczenia_iteracji = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Włączenie kamer
kamera_prawa,kamera_lewa=funkcje.wczytaj_zrodlo(1)

while True:
    # Przechwycenie pojedynczej klatki
    _, obraz_prawa = kamera_prawa.read()
    _, obraz_lewa = kamera_lewa.read()
    # Zmiana koloru na szary
    szara_prawa = cv2.cvtColor(obraz_prawa, cv2.COLOR_BGR2GRAY)
    szara_lewa = cv2.cvtColor(obraz_lewa, cv2.COLOR_BGR2GRAY)

    # Wyświetlenie obrazu z kamer
    funkcje.draw_calibration(obraz_lewa, obraz_prawa, "Obrazy")


    # Znalezienie rogow szachownicy (9,6) liczba rogow szachownicy
    czy_znaleziono_prawa, rogi_prawa = cv2.findChessboardCorners(szara_prawa, (9, 6), None)
    czy_znaleziono_lewa, rogi_lewa = cv2.findChessboardCorners(szara_lewa, (9, 6), None)

    # Jeżeli na obrazie znajduje się szachownica-dodanie punktów do tablicy
    if czy_znaleziono_prawa & czy_znaleziono_lewa:

        # Rysowanie i wyświetlenie
        obraz_lewa2=obraz_lewa.copy()
        obraz_prawa2=obraz_prawa.copy()
        cv2.drawChessboardCorners(obraz_prawa2, (9, 6), rogi_prawa, czy_znaleziono_prawa)
        cv2.drawChessboardCorners(obraz_lewa2, (9, 6), rogi_lewa, czy_znaleziono_lewa)
        funkcje.draw_calibration(obraz_lewa2,obraz_prawa2,"Szachownice")
        if cv2.waitKey(0) & 0xFF == ord('z'):  # Zapisanie 'z', Anulowanie-reszta
            t = str(i)
            print('Zapisano' + t)
            cv2.imwrite('Kalibracja/Prawa/chessboard-R' + t + '.png', obraz_prawa)
            cv2.imwrite('Kalibracja/Lewa/chessboard-L' + t + '.png', obraz_lewa)
            i = i + 1
        else:
            print('Anulowano')

    if cv2.waitKey(1) & 0xFF == ord(' '):  # Spacja - opuszczenie programu
        break

# Zakończnie programu
kamera_prawa.release()
kamera_lewa.release()
cv2.destroyAllWindows()
