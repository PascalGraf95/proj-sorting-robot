/*
Define the pins and ports of the arduino to drive a stepper
*/

// Cancels the command if another input is awaited (for example when setting the PWM duty cycle)
#define CMD_CANCEL 0xFF

// "Stop motor"
#define CMD_STOP_MOTOR 0xFF

// "Forward motor"
#define CMD_MOTOR_FORWARD 0x01

// "Backward motor"
#define CMD_MOTOR_BACKWARD 0x02


// "Set speed"
// Read the stepps to do
// Value behind 0x03 defines, how many steps the motor should do
#define CMD_SET_DATA 0x03


// "JIGGLE motor"
#define CMD_JIGGLE_MOTOR 0x04


#define CMD_


#define MODE_OFF 0
#define MODE_BACKWARD 1
#define MODE_FORWARD 2
#define MODE_JIGGLE 3
#define MODE_STEPS 4
