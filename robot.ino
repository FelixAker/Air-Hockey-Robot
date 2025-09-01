// Pin assignments for Motor 1 (X-Y combination)
#define EN1   11
#define STEP1 12
#define DIR1  13

// Pin assignments for Motor 2 (X+Y combination)
#define EN2   8
#define STEP2 9
#define DIR2  10

// Motor behavior parameters
#define MIN_STEP_DELAY 17
#define MAX_STEP_DELAY 34
#define ACCEL_DELAY 1000  // us

// Timers
unsigned long lastStepTime = 0;
unsigned long startTime;

void setup() {
  // Motor pin setup
  pinMode(STEP1, OUTPUT);
  pinMode(DIR1, OUTPUT);
  pinMode(EN1, OUTPUT);

  pinMode(STEP2, OUTPUT);
  pinMode(DIR2, OUTPUT);
  pinMode(EN2, OUTPUT);

  // Enable motors
  digitalWrite(EN1, LOW);
  digitalWrite(EN2, LOW);

  Serial.begin(9600);
  startTime = millis();
}

void step_motors(int delta1, int delta2) {
  long step_delay = MAX_STEP_DELAY;
  long t_prev = micros();

  digitalWrite(DIR1, delta1 > 0);
  digitalWrite(DIR2, delta2 > 0);

  int steps1 = abs(delta1);
  int steps2 = abs(delta2);

  while (steps1 > 0 || steps2 > 0) {
    if (steps1 > 0) digitalWrite(STEP1, HIGH);
    if (steps2 > 0) digitalWrite(STEP2, HIGH);

    delayMicroseconds(step_delay);

    if (steps1 > 0) digitalWrite(STEP1, LOW);
    if (steps2 > 0) digitalWrite(STEP2, LOW);

    delayMicroseconds(step_delay);

    if (steps1 > 0) steps1--;
    if (steps2 > 0) steps2--;

    // Accelerate gradually
    if ((micros() - t_prev) > ACCEL_DELAY) {
      t_prev = micros();
      if (step_delay > MIN_STEP_DELAY) step_delay--;
    }
  }
}

void loop() {
  if (Serial.available()) {
    String data = Serial.readStringUntil('\n');
    int comma_idx = data.indexOf(',');
    if (comma_idx > 0) {
      int delta1 = data.substring(0, comma_idx).toInt();  
      int delta2 = data.substring(comma_idx + 1).toInt(); 


      
        step_motors(delta1, delta2);
        lastStepTime = millis();
      

      startTime = millis();  // Reset idle timer
      Serial.println("OK");
    }
  }

  // Optional: Disable motors after 30s idle
  if (millis() - startTime > 30000) {
    digitalWrite(EN1, HIGH); // Disable motor driver
    digitalWrite(EN2, HIGH);
    while (1);  // Halt
  }
}
