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
uint8_t jiggletime = 50;

int delaytime = 200;

uint8_t steppsatonce = 1;

// Vaqriable TO store the stepps
int steppsData = 0;
int n_steps = 0;
byte highByte, lowByte;

unsigned long timeoutMillis = millis() + 1000;  // Set a timeout of 1000 milliseconds

// Function prototypes
void forwardstep(int delaytime, int steppsatonce);
void backwardstep(int delaytime, int steppsatonce);


void setup() {
  // Start the serial communication at 9600 baud
  Serial.begin(9600);
  // Set Velocity and Acceloration of the stepper
  stepper.setMaxSpeed(1000); // Max Velocity in [Stepps/second]
  stepper.setAcceleration(500); // Acceloration in [Stepps/second^2]
}

void loop() {
  // Check if there's any incoming data from the serial connection
  if (Serial.available() >= 2) {
    // Read the incoming byte
    incomingByte = Serial.read();
    lowByte = Serial.read();
  
    Serial.write(incomingByte);
    Serial.write(lowByte);   

    switch (incomingByte) {
      case CMD_MOTOR_FORWARD:
        Mode = MODE_FORWARD;
        break;

      case CMD_MOTOR_BACKWARD:
        Mode = MODE_BACKWARD;
        break;
        
      case CMD_SET_DATA:
        n_steps = static_cast<int>(lowByte);
        Mode = MODE_STEPS;
        break;

      case CMD_STOP_MOTOR:
        Mode = MODE_OFF;
        // Turn off the output by setting the pin LOW
        break;
        
      case CMD_JIGGLE_MOTOR:
        Mode = MODE_JIGGLE;
        break;
        
      default:
        // ignore unknown commands
        Mode = MODE_OFF; // Safety feature
        break;
    }
  }

  switch (Mode) {
    case MODE_FORWARD:
      forwardstep(delaytime, steppsatonce);
      break;
    
    case MODE_BACKWARD:
      backwardstep(delaytime, steppsatonce);
      break;
    
    case MODE_OFF:
      Mode = MODE_OFF;
      break;
    
    case MODE_JIGGLE:
      forwardstep(delaytime/2, steppsatonce);
      forwardstep(delaytime/2, steppsatonce);
      forwardstep(delaytime/2, steppsatonce);
      backwardstep(jiggletime/2, steppsatonce);
      break;
    
    case MODE_STEPS:
      // Sende den Wert von n_steps zurück an Python
      forwardstep(1, n_steps);
      Mode = MODE_OFF;
      break;
      
    default:
      // Turn off The Motor (Safety feature)
      Mode = MODE_OFF;
      break;
  }
}

// Function definitions for forwardstep and backwardstep
void forwardstep(int delaytime, int steppsatonce) {
  stepper.moveTo(stepper.currentPosition() + steppsatonce);
  stepper.runToPosition();
  // Delay
  delay(delaytime);
}

void backwardstep(int delaytime, int steppsatonce) {
  stepper.moveTo(stepper.currentPosition() - steppsatonce);
  stepper.runToPosition();
  // Delay
  delay(delaytime);
}
