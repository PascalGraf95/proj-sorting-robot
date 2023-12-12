#include "stepper.h"
#include <AccelStepper.h>


// Define a variable to store the incoming serial data
uint8_t incomingByte = 0;
uint8_t awaitingPayload = 0;

// Initialisieren des Stepper-Motors
AccelStepper stepper(1, 2, 5); // Schrittmotor angeschlossen an Pins 2 (Step) und 5 (Direction)

// Init the Motor Off
uint8_t Mode = 0;


void setup()
{
  // Start the serial communication at 9600 baud
  Serial.begin(9600);
  // Set Velocity and Acceloration of the stepper
  stepper.setMaxSpeed(1000); // Max Velocity in [Stepps/second]
  stepper.setAcceleration(500); // Acceloration in [Stepps/second^2]
}


void loop()
{
  // Check if there's any incoming data from the serial connection
  if (Serial.available() > 0)
  {
    // Read the incoming byte
    incomingByte = Serial.read();

    // If previously a command was sent that requires a payload, handle the received byte differently.
    if(awaitingPayload)
    {
      // if a "cancel command" was sent, ignore the payload and reset the "awaiting payload" state
      Serial.write(incomingByte);
      if(incomingByte == CMD_CANCEL)
      {
        awaitingPayload = 0;
      }
      else
      {
        // ...
      }
    }
    else
    {
      Serial.write(incomingByte);
      switch(incomingByte)
      {
        case CMD_START_MOTOR:
          // Turn on the output by setting the pin HIGH
          // TODO: wenn mit dem Timer gearbeitet wird, kann man ja eine duty cycle rampe fahren
          stepper.moveTo(stepper.currentPosition() + 1);
          stepper.runToPosition();
          // Delay by 1/4 of a second
          delay(250);
          Mode = MODE_ON;
          break;
        case CMD_STOP_MOTOR:
          Mode = MODE_OFF;
          // Turn off the output by setting the pin LOW
          break;
        default:
          // ignore unknown commands
          break;
      }
    }
  }

  switch(Mode){
    case MODE_ON:
      stepper.moveTo(stepper.currentPosition() + 1);
      stepper.runToPosition();
      // Delay by 1/4 of a second
      delay(250);
      break;
    case MODE_JIGGLE:
      // Jiggle Mode
      break;
    default:
      // Turn off The Motor (Safetyfeature)
      Mode = MODE_OFF;
      break;
  }
}