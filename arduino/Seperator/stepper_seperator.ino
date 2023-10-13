#include "stepper.h"

// Define the arduino pin that the stepper driver is connected to
const int pin_step = 13;
const int pin_direction = 14;

// Define a variable to store the incoming serial data
uint8_t incomingByte = 0;
uint8_t awaitingPayload = 0;


void setup()
{
  // Set the LED pin as an output
    pinMode(pin_step, OUTPUT);
    pinMode(pin_direction, OUTPUT);

    // Set "HIGH" to turn the stepper right, and "LOW" to turn left
    digitalWrite(pin_direction, HIGH);

    // Start the serial communication at 9600 baud
    Serial.begin(9600);
}

void loop(){
// Check if there's any incoming data from the serial connection
    if (Serial.available() > 0){
    // Read the incoming byte
        incomingByte = Serial.read();

// If previously a command was sent that requires a payload, handle the received byte differently.
    if(awaitingPayload){
        // if a "cancel command" was sent, ignore the payload and reset the "awaiting payload" state
            Serial.write(incomingByte);
        if(incomingByte == CMD_CANCEL){
            awaitingPayload = 0;
        }
        else {
        ToDo: read Payload
            awaitingPayload = incomingByte [2:]
        }
    }

    else
    {
        Serial.write(incomingByte);
        switch(incomingByte) {
            case CMD_START_MOTOR:
            // Turn on the output by setting the pin HIGH-
                digitalWrite(pin_step, HIGH);
                break;
            case CMD_STOP_MOTOR:
            // Turn off the output by setting the pin LOW
                digitalWrite(pin_step, LOW);
                break;
            case CMD_SET_STEPS:
            // Make a step
                digitalWrite(pin_step, HIGH);
                delay(1);
                digitalWrite(pin_step, LOW);
                delay(1);
            default:
            // ignore unknown commands
                break;
            } // switch-END
        } // else-END
    } // if-END
} // loop-END