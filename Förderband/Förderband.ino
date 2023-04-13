//  |-------------------------------------------------------------------------|
//  |                                                                         |
//  |      Programm zur Steuerung eines Förderbandes                          |
//  |                                                                         |
//  |                                                                         |
//  |                                                                         |
//  |                                                                         |
//  |                                                                         |
//  |   Poti Analog Wertebereich  0 - 1023                                    |
//  |                   Links     0 - 412        0,100,200,300,400,412        |
//  |                   Rechts  612 - 1023       612,712,812,912,1012,1023    |
//  |                                                                         |
//  |                                                                         |
//  |                                                                         |
//  |                                                                         |
//  |                                                                         |
//  |-------------------------------------------------------------------------|

#include <LiquidCrystal.h>
// LCD-Display hängt an den Pins 8,9,4,5,6,7 
LiquidCrystal lcd(8, 9, 4, 5, 6, 7 );

// Poti "maps" 
#define MotLinksMin  0
#define MotLinksMax  412

#define MotRechtsMin  612
#define MotRechtsMax  1023

#define MotStop  512

#define MotAnalogMin  140
#define MotAnalogMax  255
 



void DoMotorTest(void);
// PortPin Zuweisungen -----------------------------------------------------
// Potentiometer
int PinPoti = A15;              // Potentiometer  Analog Pin 15
const int SensorPin = 20;       // Interrupt Pin 20 Mega für HALL-Sensor
// Motor
int enA = 44;             // Analog Pin -> Motor
int in1 = 45;             // Polarität des Motor's 
int in2 = 46;              
//   in1  | in2
//    0   |  0    -> Motor steht, da gleiche polarität 
//    0   |  1    -> Motor dreht links      Geschwindigkeit hängt 
//    1   |  0    -> Motor dreht rechts     vom Analog Ausgang ab
//    1   |  1    -> Motor steht, da gleiche polarität 

// Globale Variablen ---------------------------------------------------------
// Umschaltung Auto oder manueller Betrieb 
bool hardwareControl = true;    //  => true -> Poti   false -> PC CMDs
volatile int nEdgesCount = 0; // ISR var
volatile int nSpeedCount = 0; // ISR var
double v;

//  |-------------------------------------------------------------------------|
//  |                                                                         |
//  |      S E T U P                                                          |
//  |                                                                         |
//  |                                                                         |
//  |                                                                         |
//  |                                                                         |
//  |-------------------------------------------------------------------------|
void setup() {

  //Serial.begin(9600);                           // Übertragungsgeschwindigkeit wird festgelegt
    Serial.begin(115200);                         // Übertragungsgeschwindigkeit wird festgelegt
  pinMode (SensorPin, INPUT) ;                    // Sensor Pin initialisieren
  digitalWrite(SensorPin, HIGH);                  // Activation of internal pull-up resistor
  digitalWrite(SensorPin, LOW);                   // Activation of internal pull-up resistor
  
  pinMode (enA, OUTPUT);
  pinMode (in1, OUTPUT);
  pinMode (in2, OUTPUT);
  // Nachdem die Pins als Ausgang geschaltet wurden müssen wir den Motor vorsichtshalber stoppen
  StopMotor();
  
  // Wir haben ein Zweizeilen Display a 16 Zeichen
  lcd.begin(16, 2);
  // starten links oben (0,0) 
  lcd.setCursor(0, 0);
  // und schreiben hier mal eben "Foerderband" hin 
  lcd.print("Foerderband");
  DoMotorTest();
  

// Timer 1
  noInterrupts();           // Alle Interrupts temporär abschalten
  TCCR1A = 0;
  TCCR1B = 0;

  TCNT1 = 34286;            // Timer nach obiger Rechnung vorbelegen
  TCCR1B |= (1 << CS12);    // 256 als Prescale-Wert spezifizieren
  //TCCR1B |= (1 << CS10); TCCR1B |= (1 << CS12);
  TIMSK1 |= (1 << TOIE1);   // Timer Overflow Interrupt aktivieren
  interrupts();             // alle Interrupts scharf schalten


  
}

//  |-------------------------------------------------------------------------|
//  |                                                                         |
//  |      L O O P                                                            |
//  |                                                                         |
//  |                                                                         |
//  |                                                                         |
//  |                                                                         |
//  |                                                                         |
//  |                                                                         |
//  |                                                                         |
//  |                                                                         |
//  |-------------------------------------------------------------------------|

#define SerialDelay 3000
void loop() {
  String CmdString;
  static int nPotiSave; 
  static int nSpeed = 1100;

  static unsigned long lLastTime = 0;   // Speicher für den Zeitstempel
  unsigned long lCurTime;
  double v;                             // aktuelle Geschwindigkeit 

   v = (nSpeedCount*((3.14*51)/8))/3;                 //in mm/s              
   //v = (nSpeedCount*((3.14*5.1)/8))/3;              //in cm/s    

   display_V(v);
  // CMD über serielle Schnittstelle polen
  CmdString = receiveCmdString();
  CmdString.trim();

//  |-------------------------------------------------------------------------|
//  |       Wenn die Länge des CMD Strings > 0 ist haben wir ein Kommando     |
//  |       empfangen.                                                        |                         
//  |-------------------------------------------------------------------------|    
    if(CmdString.length() ) {
      hardwareControl = false;                  // => PC Mode aktiv 
      DoCmd(CmdString);               // Kommandostring auswerten
      Serial.println("Serial Input");
    }

//  |-------------------------------------------------------------------------|
//  |       Wenn der Wert des Potis unterschiedlich ist ........              |
//  |                                                                         |                         
//  |-------------------------------------------------------------------------|
    if(hardwareControl) {                      // Handbetrieb
      DoPoti();
    }
    
//  |-------------------------------------------------------------------------|
//  |       Ausgabe der Geschwindikgeit                                       |
//  |                                                                         |                         
//  |-------------------------------------------------------------------------|
  lCurTime = millis();
  if(lCurTime - lLastTime > SerialDelay)
  {
     /*
      Serial.print("@");
      Serial.print(nSpeedCount);
      Serial.print("A");
      Serial.print(v);
      Serial.println("B");
      */

      lLastTime = lCurTime;          // Zeitstempel der letzten Übertragung sichern
  }

}

//  |-------------------------------------------------------------------------|
//  |                                                                         |
//  |       LCD Ausgbae                                                       |
//  |                                                                         |
//  |                                                                         |
//  |-------------------------------------------------------------------------|
void display_V(double Vist) 
{
  lcd.setCursor(0, 0);
  lcd.print("Foerderband");
  lcd.setCursor(0, 1);
  lcd.print("V: ");
  lcd.print(Vist);
  //Serial.println(v);
  lcd.print(" mm/s               ");
}


//  |-------------------------------------------------------------------------|
//  |                                                                         |
//  |       DoCmd                                                             |
//  |                                                                         |
//  |       - Kommandos auswerten                                             |
//  |                                                                         |
//  |                                                                         |
//  |                                                                         |
//  |                                                                         |
//  |                                                                         |
//  |                                                                         |
//  |-------------------------------------------------------------------------|

void DoCmd(String sCmd) {
static int nCmdSpeed = MotStop;                     // lokale Variable für die Geschwindigkeit
static bool boMotRechts = true;                     // starten immer mit rechts 

char bMotCmd;
  
  if(sCmd.length()){
    Serial.println("sCmd = " + sCmd);
    bMotCmd = sCmd[0];                              // 4. Zeichen als Kommando BYTE übernehmen (für switch case)
    
    
    // a = Umschaltung auf Poti-Modus
    if(sCmd == "a" ){
         hardwareControl = true;                               // => Poti Mode aktiv
     }


    switch(bMotCmd) {
//  |---------------------------------------------------------------------------------------------------|
      case '0':                                                         // 0 -> Motor stop    
              Serial.println("MotorStop");
              nCmdSpeed = MotStop;
              StopMotor();
              break;    
//  |---------------------------------------------------------------------------------------------------|
      case '1':                                                         // 1 -> starte Motor      
              Serial.println("MotorStart");      
              if(boMotRechts) {
                  if(nCmdSpeed == MotStop) {
                    nCmdSpeed = MotRechtsMin;
                  }
                  MotorRechts();                                        // Motorpolarität auf "Rechts" stellen
                  SetSpeedRechts(nCmdSpeed); 
 
              } else {
                  if(nCmdSpeed == MotStop) {
                    nCmdSpeed = MotLinksMax;
                  }
                  MotorLinks();                                         // Motorpolarität auf "Links" stellen
                  SetSpeedLinks(nCmdSpeed);  
              }
              break;
//  |---------------------------------------------------------------------------------------------------| 
      case '+':                                                         // + -> Motor schnell
              Serial.println("MotorSchnell");
              if(boMotRechts) {
                  nCmdSpeed = GetIncreaseSpeedValueRechts(nCmdSpeed);
                  SetSpeedRechts(nCmdSpeed);                            // und Geschwindigkeit über Potiwert einstellen    
              } else {
                  nCmdSpeed = GetIncreaseSpeedValueLinks(nCmdSpeed);
                  SetSpeedLinks(nCmdSpeed);                             // und Geschwindigkeit über Potiwert einstellen                  
              }
              break;
//  |---------------------------------------------------------------------------------------------------|
      case '-':                                                         // - -> Motor langsam    
              Serial.println("MotorLangsam");
              if(boMotRechts) {
                  nCmdSpeed = GetDecreaseSpeedValueRechts(nCmdSpeed);
                  SetSpeedRechts(nCmdSpeed);                            // und Geschwindigkeit über Potiwert einstellen    
              } else {
                  nCmdSpeed = GetDecreaseSpeedValueLinks(nCmdSpeed);
                  SetSpeedLinks(nCmdSpeed);                             // und Geschwindigkeit über Potiwert einstellen                  
              }
              break;
//  |---------------------------------------------------------------------------------------------------|              
      case 'l':                                                         // l -> Motor links
              Serial.println("MotorLinks");
              boMotRechts = false;                                      // Merker setzen
              nCmdSpeed = MotLinksMax;
              MotorLinks();                                             // Motorpolarität auf "Links" stellen
              MotorStop();                                              // und Geschwindigkeit über Potiwert einstellen
              break;
//  |---------------------------------------------------------------------------------------------------|
      case 'r':                                                         // r -> Motor rechts
              Serial.println("MotorRechts");
              boMotRechts = true;                                       // Merker setzen
              nCmdSpeed = MotRechtsMin;
              MotorRechts();                                            // Motorpolarität auf "Rechts" stellen
              MotorStop();                                              // und Geschwindigkeit über Potiwert einstellen  
              break;
 //  |---------------------------------------------------------------------------------------------------|              
      default:  
              hardwareControl = true;                                             // => Poti Mode aktiv
              break;
    
    } // end switch case

  } // keine Daten in Cmd-String vorhanden
          
}

//  |-------------------------------------------------------------------------|
//  |                                                                         |
//  |       DoPoti                                                            |
//  |                                                                         |
//  |       - Potentiometerwert einlesen und in Abhängikeit vom Potiwert      |
//  |         den Motor mit übergebener Geschwindikeit ansteuern.             |
//  |                                                                         |
//  |                                                                         |
//  |                                                                         |
//  |                                                                         |
//  |                                                                         |
//  |-------------------------------------------------------------------------|
void DoPoti(void)
{
int nPotiValue;                     // lokale Variable für den Potiwert
  
  nPotiValue = analogRead(PinPoti); // Mittelabgriff Poti lesen

  //Serial.print("nPotiValue = ");
  //Serial.println(nPotiValue);
  
  if (nPotiValue < MotLinksMax) {  
    //Serial.println(" MotorLinks"); 
    MotorLinks();                   // Motorpolarität auf "Links" stellen
    SetSpeedLinks(nPotiValue);      // und Geschwindigkeit über Potiwert einstellen
  } else if(nPotiValue > MotRechtsMin){
    //Serial.println(" MotorRechts"); 
    MotorRechts();                  // Motorpolarität auf "Rechts" stellen
    SetSpeedRechts(nPotiValue);     // und Geschwindigkeit über Potiwert einstellen
   
  } else {                         
    //Serial.println(" StopMotor"); 
    StopMotor();                    // Motor aus
  }

}

//  |-------------------------------------------------------------------------|
//  |       SetSpeedRechts                                                    |
//  |-------------------------------------------------------------------------|
void SetSpeedRechts(int nSpeed)
{
  int nMotorSpeed;
  nMotorSpeed = map(nSpeed,MotRechtsMin,MotRechtsMax,MotAnalogMin,MotAnalogMax);         
  analogWrite(enA, nMotorSpeed);                    // losfahren des Motors
}
//  |-------------------------------------------------------------------------|
//  |       SetSpeedLinks                                                     |
//  |-------------------------------------------------------------------------|
void SetSpeedLinks(int nSpeed)
{
  int nMotorSpeed;
  nMotorSpeed = map(nSpeed,MotLinksMax,MotLinksMin,MotAnalogMin,MotAnalogMax);           
  analogWrite(enA, nMotorSpeed);                    // losfahren des Motors
}
//  |-------------------------------------------------------------------------|
//  |       StopMotor                                                         |
//  |-------------------------------------------------------------------------|
void StopMotor()
{
  //Serial.println("StopMotor");
  MotorStop();                                      // Über die polarität den Motor ausschalten
}



//  |-------------------------------------------------------------------------|
//  |                   Links     0 - 412        0,100,200,300,400,412        |
//  |                   Rechts  612 - 1023       612,712,812,912,1012,1023    |
//  |-------------------------------------------------------------------------|
//  |       Geschwindigkeitswert bestimmen "schnell"    rechts                |
//  |-------------------------------------------------------------------------|
int GetIncreaseSpeedValueRechts(int nCurrentSpeed)
{
  //Serial.println("GetIncreaseSpeedValueRechts");
  if(nCurrentSpeed >= MotRechtsMin && nCurrentSpeed <= MotRechtsMax -100){
        nCurrentSpeed += 100;
  } else {
    nCurrentSpeed = MotRechtsMax;
  }
  //Serial.println(nCurrentSpeed);
return nCurrentSpeed;   
}


//  |-------------------------------------------------------------------------|
//  |       Geschwindigkeitswert bestimmen "langsam"    rechts                |
//  |-------------------------------------------------------------------------|
int GetDecreaseSpeedValueRechts(int nCurrentSpeed)
{
  //Serial.println("GetDecreaseSpeedValueRechts");
  if(nCurrentSpeed >= MotRechtsMin+100 && nCurrentSpeed <= MotRechtsMax){
        nCurrentSpeed -= 100;
  } else {
    nCurrentSpeed = MotRechtsMin;
  }
  //Serial.println(nCurrentSpeed);
return nCurrentSpeed;   
}

//  |-------------------------------------------------------------------------|
//  |       Geschwindigkeitswert bestimmen "schnell"    links                 |
//  |-------------------------------------------------------------------------|
int GetIncreaseSpeedValueLinks(int nCurrentSpeed)
{
  //Serial.println("GetIncreaseSpeedValueLinks");
  if(nCurrentSpeed >= MotLinksMin+100 && nCurrentSpeed <= MotLinksMax){
        nCurrentSpeed -= 100;
  } else {
    nCurrentSpeed = 0;
  }
  //Serial.println(nCurrentSpeed);
return nCurrentSpeed;   
}

//  |-------------------------------------------------------------------------|
//  |       Geschwindigkeitswert bestimmen "langsam"    links                 |
//  |-------------------------------------------------------------------------|
int GetDecreaseSpeedValueLinks(int nCurrentSpeed)
{
  if(nCurrentSpeed >= MotLinksMin && nCurrentSpeed <= MotLinksMax-100){
        nCurrentSpeed += 100;
  } else {
    nCurrentSpeed = MotLinksMax;
  }
 //Serial.println(nCurrentSpeed);
return nCurrentSpeed;   
}


//  |-------------------------------------------------------------------------|
//  |                                                                         |
//  |       receiveCmdString                                                  |
//  |       CMD String empfangen                                              |
//  |-------------------------------------------------------------------------|
String receiveCmdString(void) 
{
String IncomingString = ""; 
 // Wenn Zeichen vorhanden ist
 if (Serial.available() > 0) {                  
    IncomingString = Serial.readStringUntil('\n');
 }
 // Empfangenen String zurückgeben
 return(IncomingString);
}


//  |-------------------------------------------------------------------------|
//  |       MotorStop()                                                       |
//  |-------------------------------------------------------------------------|
void MotorStop()
{
  analogWrite(enA, 0);                                // Motor aus
}
//  |-------------------------------------------------------------------------|
//  |       MotorLinks()                                                      |
//  |-------------------------------------------------------------------------|
void MotorLinks()
{
  digitalWrite(in1,HIGH);              
  digitalWrite(in2,LOW);
}
//  |-------------------------------------------------------------------------|
//  |       MotorRechts()                                                     |
//  |-------------------------------------------------------------------------|
void MotorRechts()
{
  digitalWrite(in1,LOW);      
  digitalWrite(in2,HIGH);  
}


//  |-------------------------------------------------------------------------|
//  |                                                                         |
//  |       Timer Interrupt                                                   |
//  |                                                                         |
//  |       Aufruf alle 0,5 Sekunden                                          |
//  |                                                                         |
//  |                                                                         |
//  |                                                                         |
//  |                                                                         |
//  |                                                                         |
//  |                                                                         |
//  |-------------------------------------------------------------------------|
// Hier kommt die selbstdefinierte Interruptbehandlungsroutine 
// für den Timer Overflow
ISR(TIMER1_OVF_vect)        
{ 
  static int nIsrTimerCount = 0;
  TCNT1 = 34286;             // Zähler erneut vorbelegen-> Nächster Timer Interrupt in 0,5 Sekunden
 
 // Aufruf alle 3 Sekunden
 if(nIsrTimerCount++ >= 6) {
  detachInterrupt(digitalPinToInterrupt(SensorPin));
  nSpeedCount = nEdgesCount;
  nEdgesCount = 0;
  attachInterrupt(digitalPinToInterrupt(SensorPin), HandlerISR_HalSensor, RISING);
 //  Serial.print("nSpeedCount -> " );
 //  Serial.println(nSpeedCount);
  
  nIsrTimerCount = 0; 
 }
  
   
}

//  |-------------------------------------------------------------------------|
//  |                                                                         |
//  |       ISR                                                               |
//  |                                                                         |
//  |           Zählt rising edges vom hal sensor                             |
//  |                                                                         |
//  |                                                                         |
//  |                                                                         |
//  |                                                                         |
//  |-------------------------------------------------------------------------|
#define TriggerDelay 500
unsigned long lLastTimestamp = 0;   // Zeitstempel

void HandlerISR_HalSensor(void)
{
unsigned long lCurTime;
 

  lCurTime = micros();
  /*
  Serial.print( (lLastTimestamp));
  Serial.print("\t ");
  Serial.print( (lCurTime));
  Serial.print("\t ");
  Serial.println( (lCurTime-lLastTimestamp) / 1000);
  lLastTimestamp = lCurTime; 
*/
  
  if(lCurTime-lLastTimestamp > TriggerDelay){
      nEdgesCount++;                // Nur zählen, wenn Entprellzeit abgelaufen ist
/*
      Serial.print(nEdgesCount);
      Serial.print(": ");
      Serial.print( (lCurTime-lLastTimestamp) );
      Serial.println(" ");
*/
      lLastTimestamp = lCurTime;    // Zeitstempel der letzten "gültigen" Zählung sichern
  }
  
}

void DoMotorTest(void)
{
  unsigned int nSpeed;
  unsigned int nValue;
  int i;
  
  RampeRechts();
  delay(500);
  RampeLinks();
  
}

void RampeRechts(void)
{
unsigned int nSpeed;
unsigned int nValue;
int i;
   MotorRechts(); // Polarität
   
   for(i=MotRechtsMin;i<=MotRechtsMax;i++) {
    nValue = i;
    nSpeed=map(nValue,MotRechtsMin,MotRechtsMax,MotAnalogMin,MotAnalogMax);          // Rechts
    /*
    delay(5);
    Serial.print("nValue = ");
    Serial.print(nValue);
    Serial.print(" => ");
    Serial.print("nSpeed = ");
    Serial.print(nSpeed);
    Serial.println(" MAX rechts");
    */
    analogWrite(enA, nSpeed);                // losfahren des Motors
    delay(1);
  }
  //Serial.println(" *************************************\n\n");
  
  //delay(2000);
 for(i=MotRechtsMax;i>=MotRechtsMin;i--) {
    nValue = i;
    nSpeed=map(nValue,MotRechtsMin,MotRechtsMax,MotAnalogMin,MotAnalogMax);          // Rechts
    /*
    delay(5);
    Serial.print("nValue = ");
    Serial.print(nValue);
    Serial.print(" => ");
    Serial.print("nSpeed = ");
    Serial.print(nSpeed);
    Serial.println(" MIN rechts");    
    */
    analogWrite(enA, nSpeed);                // losfahren des Motors
    delay(1);

  }
  //Serial.println(" *************************************\n\n");
  
  MotorStop();
  //SetSpeedLinks(000);
}


void RampeLinks(void)
{
byte nSpeed;
unsigned int nValue;
int i;
   MotorLinks(); // Polarität
   
   for(i=MotLinksMax;i>=1;i--) {
    nValue = i;
    nSpeed=map(nValue,MotLinksMax,MotLinksMin,MotAnalogMin,MotAnalogMax);           // Links
    delay(1);
    /*
    Serial.print("nValue = ");
    Serial.print(nValue);
    Serial.print(" => ");
    Serial.print("nSpeed = ");
    Serial.print(nSpeed);
    Serial.println(" MAX links");
    */
    analogWrite(enA, nSpeed);                // losfahren des Motors
    //delay(100);

  }
  //Serial.println(" *************************************\n\n");
  //delay(2000);
  
  
 for(i=1;i<=MotLinksMax;i++) {
    nValue = i;
    nSpeed=map(nValue,MotLinksMax,MotLinksMin,MotAnalogMin,MotAnalogMax);           // Links
    delay(2);
 /* 
    Serial.print("nValue = ");
    Serial.print(nValue);
    Serial.print(" => ");
    Serial.print("nSpeed = ");
    Serial.print(nSpeed);
    Serial.println(" MIN links");
    */
    analogWrite(enA, nSpeed);                // losfahren des Motors
    //delay(100);

  }
  //Serial.println(" *************************************\n\n");
  
  MotorStop();
  //SetSpeedLinks(000);
}
