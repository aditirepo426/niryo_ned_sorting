// Simple serial-controlled conveyor controller.
// Commands from Python: START, STOP.
// IR sensor event: sends OBJECT_DETECTED once per object.

const int MOTOR_EN = 5;
const int MOTOR_IN1 = 6;
const int MOTOR_IN2 = 7;
const int IR_PIN = 2;

bool conveyorRunning = false;
bool objectLatched = false;

void startConveyor() {
  digitalWrite(MOTOR_IN1, HIGH);
  digitalWrite(MOTOR_IN2, LOW);
  analogWrite(MOTOR_EN, 180);
  conveyorRunning = true;
}

void stopConveyor() {
  analogWrite(MOTOR_EN, 0);
  digitalWrite(MOTOR_IN1, LOW);
  digitalWrite(MOTOR_IN2, LOW);
  conveyorRunning = false;
}

void setup() {
  pinMode(MOTOR_EN, OUTPUT);
  pinMode(MOTOR_IN1, OUTPUT);
  pinMode(MOTOR_IN2, OUTPUT);
  pinMode(IR_PIN, INPUT);
  Serial.begin(115200);
  stopConveyor();
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "START") startConveyor();
    else if (cmd == "STOP") stopConveyor();
  }

  bool detected = digitalRead(IR_PIN) == LOW; // change if sensor logic is inverted
  if (detected && !objectLatched) {
    objectLatched = true;
    stopConveyor();
    Serial.println("OBJECT_DETECTED");
  }
  if (!detected) objectLatched = false;
}
