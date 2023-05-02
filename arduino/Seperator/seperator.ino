#include "commands.h"

// Define the pin that the transistor is connected to
const int output_pin = 13;

// Define a variable to store the incoming serial data
uint8_t incomingByte = 0;
uint8_t awaitingPayload = 0;


void setup()
{
  // Set the LED pin as an output
  pinMode(output_pin, OUTPUT);

  // Start the serial communication at 9600 baud
  Serial.begin(9600);
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
          digitalWrite(output_pin, HIGH);
          break;
        case CMD_STOP_MOTOR:
          // Turn off the output by setting the pin LOW
          digitalWrite(output_pin, LOW);
          break;
        default:
          // ignore unknown commands
          break;
      }
    }
  }
}