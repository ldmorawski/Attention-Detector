#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <string> 
#include <conio.h>
#include <iostream> 
using namespace cv;
using namespace std;
//Deklaracja sta�ej ilo�ci klatek, z kt�rej liczona jest �rednia uwaga
float nrOfPeople = 0;
int cameraID = 0;
//Funkcja �aduj�ca parametry programu
void loadParameters()
{
	char oneCam = 0;	//Zmienna przechowuj�ca dane o kamerce
	//Flaga stanowi�ca o tym, czy prawid�owo wprowadzono dane
	bool correctInput = false;
	cout << "Enter amount of people: ";
	//Miejce na podanie ilo�ci uczestnik�w przez u�ytkownika programu
	cin >> nrOfPeople;	
	while (!correctInput)
	{
		cout << "Do you have more, than one camera-type device? (Y/N)\n";
		cin >> oneCam;
		//Zamiana wprowadzonego znaku na du�� liter�
		oneCam = toupper(oneCam);
		if (oneCam == 'Y')
		{
			correctInput = true;

			//Program pyta czy pod��czono wi�cej, ni� jedn� kamer�
			cout << "which camera do you want to use?" << endl;
			cout << " 0 - built-in camera (if you have one)";
			cout << " or ext. camera 1 " << endl;
			cout << "1 - ext. camera 1 or 2 \n";
			cout << "2 - ext.camera 2 or 3, etc." << endl;
			cin >> cameraID;
			//Zabezpieczenie przed wprowadzeniem z�ego ID kamery
			cameraID = cameraID % 5;	
		}
		else if (oneCam == 'N')
		{
			correctInput = true;
			cameraID = 0;
		}
		else
			cout << "Error: Wrong input data!" << endl;
	}
}
//Funkcja zliczaj�ca �redni� arytmetyczn� el. tablicy
float countAverageValue(int tab[], int n)
{
	float sum = 0;
	for (int i = 0; i < n; i++)
		sum += tab[i];
	return sum / n;
}
//Funkcja zmieniaj�ca kolor paska uwagi w zale�no�ci od poziomu uwagi
Scalar getBarColor(int percent)
{
	Scalar color = CV_RGB(0, 0, 0);
	if (percent <= 100 && percent >= 67)
		color = CV_RGB(0, 255, 0);
	else if (percent < 67 && percent > 34)
		color = CV_RGB(255, 255, 0);
	else if (percent <= 33)
		color = CV_RGB(255, 0, 0);
	return color;
}
int main()
{
	//Sta�a liczba element�w tablicy do kt�rej wrzucane s� dane
	const int nrOfFrames = 10;
	loadParameters();					//�adowanie parametr�w
	//Maska do wyostrzenia obrazu wej�ciowego
	Mat kernel = (Mat_<double>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	Mat frame;			//Zmienna przechowuj�ca klatk� wideo 
	Mat preview;		//Zmienna przechowuj�ca okno z wynikami
	//Przygotowanie szablonu okna pod wy�wietlanie wynik�w
	Mat GUItemplate = Mat(200, 700, CV_8UC3, Scalar(0,0,0));
	putText(GUItemplate, "Attention:   ", Point(5, 50),
		FONT_HERSHEY_SIMPLEX, 2, CV_RGB(255, 255, 255), 3);
	putText(GUItemplate, "NO ", Point(5, 120),
		FONT_HERSHEY_SIMPLEX, 2, CV_RGB(255, 255, 255), 3);
	putText(GUItemplate, "FULL ", Point(505, 120),
		FONT_HERSHEY_SIMPLEX, 2, CV_RGB(255, 255, 255), 3);
	putText(GUItemplate, "To quit program, hold 'q' key.",
		Point(10, 195), FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 255, 255), 2);
	vector<Rect> faces;			//Zmienna do przechowywania wykrytych twarzy
	int attention[nrOfFrames];	//Tablica przechowuj�ca informacje o uwadze
	int percentage = 0;		//Zmienna przechowuj�ca procentow� warto�� uwagi
	//Tablica musi zosta� wype�niona, do wyliczenia �redniej
	//Flaga kontroluj�ca, czy za�adowano pierwsze 10 klatek
	bool dataLoaded = false;
	//ustawienie domy�lnej warto�ci dla czu�o�ci klasyfikatora
	int Sensitivity = 7;

	//Instancja klasy uruchamiaj�cej pobieranie obrazu
	VideoCapture capture = VideoCapture(cameraID);
	//Zmienne przechowuj�ce wielko�� ramki obrazu
	int W = 800; int H = 600;		
	//Zmiana rozdzielczo�ci obrazu
	capture.set(CV_CAP_PROP_FRAME_WIDTH, W);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, H);	
	//Zdefiniowanie wielko�ci ramki pobieranego obrazu
	Size frameSize = Size(W, H);
	//Przygotowanie okna do wy�wietlania aktualnej klatki
	//Utworzenie instancji klasy wykrywaj�cej twarz
	CascadeClassifier cl_face;	
	//�adowanie klasyfikatora do wykrywania twarzy
	cl_face.load("haarcascades\\haarcascade_frontalface_default.xml");

	//Je�li nie uda si� za�adowa�, zwr�� komunikat i zamknij program
	if (!cl_face.load("haarcascades\\haarcascade_frontalface_default.xml"))
	{
		cout << "Unable to load classifier: ";
		cout << "\"haarcascade_frontalface_default.xml\" " << endl;
		waitKey(5000);
		return -1;
	}
	//Inicjalizacja okna
	const string window_name = "Attention Detector";
	namedWindow(window_name, CV_WINDOW_AUTOSIZE);
	//Informacja o rozpocz�ciu oblicze�
	cout << "Processing has begun..." << endl;
	for (int i = 0; i < nrOfFrames; i++)
	{
		//Skopiowanie szablonu do obrazu wynikowego
		GUItemplate.copyTo(preview);
		capture >> frame;						//Akizycja obrazu
		filter2D(frame, frame, -1, kernel);		//Poprawa jako�ci klatki
		//Wykrycie twarzy klasyfikatorem
		cl_face.detectMultiScale(frame, faces, 1.1, Sensitivity);
		//Uwaga jest r�wna ilo�ci wykrytych twarzy
		attention[i] = faces.size();	
		//Gdy i == 9 tzn. �e tablica zosta�a wype�niona
		//Nast�puje zmiana flagi i zerowanie iteratora licz�cego klatki
		//i = -1 poniewa� po obrocie p�tli nast�puje i++
		if (i == 9)
		{
			i = -1;
			dataLoaded = true;
		}
		if (dataLoaded)
		{
			//Liczenie procent�w
			percentage = countAverageValue(attention, nrOfFrames) / nrOfPeople * 100;
			//Zmienna do przechowywania liczby procent�w
			char text[5] = {};
			//Zabezpieczenie przed wy�wietleniem za du�ego wyniku
			if (percentage >= 100)
				percentage = 100;
			//Funkcja konwertuj�ca int do zmiennej string (lub char*[] )
			_itoa_s(percentage,text,10);
			//Umieszczenie znaku procenta w �a�cuchu z liczb� procent�w

			if (percentage == 100)
				text[3] = '%';
			else if (percentage == 0)
				text[1] = '%';
			else
				text[2] = '%';
			//Zmienna do przechowania koloru paska
			Scalar barColor = getBarColor(percentage);
			//Pasek ma d�ugo�� 400px, po 4px na ka�dy punkt procentowy
			int barPixels = percentage * 4 + 100;
			//Umieszczenie procentowej uwagi na ekranie
			putText(preview, text, Point(300, 50),
				FONT_HERSHEY_SIMPLEX, 2, CV_RGB(255, 255, 255), 2);
			Rect percentageBar = Rect(Point(100, 120), Point(barPixels, 80));
			rectangle(preview, percentageBar, barColor, CV_FILLED);
			//Wy�wietlenie obrazu wynikowego
			imshow(window_name, preview);
		}
		//Wci�ni�cie przycisku 'q' ko�czy dzia�anie programu
		//Wyj�� z programu mo�na tylko wtedy, gdy tablica jest wype�niona
		if (dataLoaded && waitKey(1) == 'q')
		{
			capture.release();			//Zwolnienie uchwytu kamery
			//Komunikat o zako�czeniu pracy programu
			cout << "The program will close automatically in 9";
			for (int i = 8; i >= 0; i--)
			{
				waitKey(1000);
				// \b - usuwanie znaku. Zast�pienie go nowym numerem
				cout << "\b" << i;	
			}
			destroyWindow(window_name); //Zamkni�cie okna
			return 1;					//Wyj�cie z programu
		}
	}
}