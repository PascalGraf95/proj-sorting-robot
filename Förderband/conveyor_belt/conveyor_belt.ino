//  |-------------------------------------------------------------------------|
//  |      Conveyor Belt Script                                               |
//  |      by Pascal Graf - last edited 2023                                  |
//  |                                                                         |
//  |   Poti Analog Values        0 - 1023                                    |
//  |                   Links     0 - 412        0,100,200,300,400,412        |
//  |                   Rechts  612 - 1023       612,712,812,912,1012,1023    |
//  |-------------------------------------------------------------------------|

#include <LiquidCrystal.h>

// 0 - 412
const int motorLeftMinValue = 0;
const int motorLeftSlowValue = 206;
const int motorLeftFastValue = 412;

// 612 - 1023
const int motorRightMinValue = 612;
const int motorRightSlowValue = 818;
const int motorRightFastValue = 1023;

const int motorStopValue = 512;

const int motorAnalogMinValue = 140;
const int motorAnalogMaxValue = 255;

const int serialDelay = 3000;

int currentMotorDirection = 0;

LiquidCrystal lcd(8, 9, 4, 5, 6, 7 );
int potentiometerPin = 15; // Potentiometer  Analog Pin 15
// motorPins
int enA = 44;  // Analog Pin -> Motor
int in1 = 45;  // Polarität des Motor's 
int in2 = 46;  
            
//   in1  | in2
//    0   |  0    -> Motor steht, da gleiche polarität 
//    0   |  1    -> Motor dreht links      Geschwindigkeit hängt 
//    1   |  0    -> Motor dreht rechts     vom Analog Ausgang ab
//    1   |  1    -> Motor steht, da gleiche polarität 

// Auto or manual control 
bool hardwareControl = true;    //  => true -> Poti   false -> PC CMDs

//  |-------------------------------------------------------------------------|
//  |      S E T U P                                                          |
//  |-------------------------------------------------------------------------|
void setup() 
{
  Serial.begin(9600);
  
  pinMode (enA, OUTPUT);
  pinMode (in1, OUTPUT);
  pinMode (in2, OUTPUT);
  
  // Stop the motor
  stopMotor();
  
  lcd.begin(16, 2);
  lcd.setCursor(0, 0);
  lcd.print("Conveyor Belt");

  // Execute a motor test
  executeMotorTest();

  // Interrupts
  /*
  noInterrupts();           // deactivate Interrupts
  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1 = 34286;            // Timer nach obiger Rechnung vorbelegen
  TCCR1B |= (1 << CS12);    // 256 als Prescale-Wert spezifizieren
  TIMSK1 |= (1 << TOIE1);   // Timer Overflow Interrupt aktivieren
  interrupts();             // activate Interrupts
  */
}

//  |-------------------------------------------------------------------------|
//  |      L O O P                                                            |
//  |-------------------------------------------------------------------------|
void loop() 
{
  String cmdString;
  // Receive a new command string via serial connection
  cmdString = receiveCommandString();
  cmdString.trim();
  
  // If a new command has been received turn off hardware control and execute command
  if(cmdString.length()) 
  {
    hardwareControl = false;
    executeCommand(cmdString);
  }
  
  // Otherwise, if no serial command has been received yet, execute the potentiometer value
  if(hardwareControl) 
  { 
    executePotentiometerValue();
  }
}

void executeCommand(String cmdString) 
{
  char motorCommandByte = cmdString[0];
  switch(motorCommandByte) 
  {
    // Switch back to manual control
    case 'a':
      hardwareControl = true;            
    // Stop Motor
    case '0':
      stopMotor();
      break;
    // Motor Right Fast
    case '+':
      setMotorSpeedRight(motorRightFastValue);
      break;
    // Motor Right Slow
    case 'r':
      setMotorSpeedRight(motorRightSlowValue);
      break;
    // Motor Left Fast
    case '-':
      setMotorSpeedLeft(motorLeftFastValue);
      break;
    // Motor Left Slow
    case 'l':
      setMotorSpeedLeft(motorLeftSlowValue);
      break;
  }      
}

void executePotentiometerValue(void)
{
  int nPotentiometerValue = analogRead(potentiometerPin); // Mittelabgriff Poti lesen
  
  if (nPotentiometerValue < motorLeftFastValue) 
  {  
    setMotorSpeedLeft(nPotentiometerValue);
  } 
  else if(nPotentiometerValue > motorRightMinValue)
  {
    setMotorSpeedRight(nPotentiometerValue);
  } 
  else 
  {                         
    stopMotor();
  }
}

void setMotorSpeedRight(int commandValue)
{
  if(currentMotorDirection != 1)
  {
    digitalWrite(in1,LOW);      
    digitalWrite(in2,HIGH);  
    currentMotorDirection = 1;  
  }

  int nMotorSpeed = map(commandValue, motorRightMinValue, motorRightFastValue, motorAnalogMinValue, motorAnalogMaxValue);           
  analogWrite(enA, nMotorSpeed);
}

void setMotorSpeedLeft(int commandValue)
{
  if(currentMotorDirection != -1)
  {
    digitalWrite(in1,HIGH);              
    digitalWrite(in2,LOW);
    currentMotorDirection = -1;  
  }
  int nMotorSpeed = map(commandValue, motorLeftMinValue, motorLeftFastValue, motorAnalogMinValue, motorAnalogMaxValue);           
  analogWrite(enA, nMotorSpeed);
}

void stopMotor()
{
  analogWrite(enA, 0);
  currentMotorDirection = 0;
}

String receiveCommandString() 
{
  String incomingString = ""; 
  if (Serial.available() > 0) {                  
      incomingString = Serial.readStringUntil('\n');
  }
  return(incomingString);
}

void executeMotorTest(void)
{ 
  setMotorSpeedRight(motorRightFastValue);
  delay(1000);
  stopMotor();
  delay(1000);
  setMotorSpeedLeft(motorLeftFastValue); 
  delay(1000);
  stopMotor();  
}


