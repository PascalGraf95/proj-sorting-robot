/*
Define the pins and ports of the arduino to drive a stepper
*/

// Cancels the command if another input is awaited (for example when setting the PWM duty cycle)
#define CMD_CANCEL 0xFF

// "Start motor"
// enables the PWM controller to supply the motor
#define CMD_START_MOTOR 0x01

// "Stop motor"
// disables the PWM controller
#define CMD_STOP_MOTOR 0x02

// "Set speed"
// Sets the PWM duty cycle. This command has to be followed by the PWM duty cycle in % (1 byte)
// Example: 0x03 0x20   => set 32% duty cycle (0x20 = 32)
#define CMD_SET_DATA 0x03

#define CMD_SET_JIGGLE 0x04

#define MODE_OFF 0
#define MODE_ON 1
#define MODE_JIGGLE 2
