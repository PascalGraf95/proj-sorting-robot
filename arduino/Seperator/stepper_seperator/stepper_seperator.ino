#include "stepper.h"
#include <AccelStepper.h>


// Define a variable to store the incoming serial data
uint8_t incomingByte = 0;
uint8_t awaitingPayload = 0;

// Initialize the Stepper-Motor
// Stepper- Step connected at Pin 2 and Direction to Pin 5
AccelStepper stepper(1, 2, 5); 

// Init the Motor in Off Mode
uint8_t Mode = 0;

// Define the Offtime between all Impulses in Jigglemode
uint8_t jiggletime = 50; (ms)



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
          Mode = MODE_ON;
          break;
        case CMD_STOP_MOTOR:
          Mode = MODE_OFF;
          // Turn off the output by setting the pin LOW
          break;
        case CMD_SET_JIGGLE:
          Mode = MODE_JIGGLE;
          break;        
        default:
          // ignore unknown commands
          // Mode = MODE_OFF; Safetyfeature
          break;
      }
    }
  }

  switch(Mode){
    case MODE_ON:
      forwardstep(250,2);
      break;
    case MODE_ON:
      Mode = MODE_OFF;
      break;
    case MODE_JIGGLE:
      forwardstep(jiggletime);
      forwardstep(jiggletime);
      backwardstep(jiggletime);
      forwardstep(jiggletime);  
      backwardstep(jiggletime);         
      break;
    default:
      // Turn off The Motor (Safetyfeature)
      Mode = MODE_OFF;
      break;
  }
}


void forwardstep(int delaytime = 250, int steppsatonce=1)
{
  stepper.moveTo(stepper.currentPosition() + steppsatonce);
  stepper.runToPosition();
  // Delay 
  delay(delaytime);
}

void backwardstep(int delaytime = 250, int steppsatonce = 1)
{
  stepper.moveTo(stepper.currentPosition() - steppsatonce);
  stepper.runToPosition();
  // Delay 
  delay(delaytime);
}


