#define LED 4

int incomingByte;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(LED, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available() > 0){
    incomingByte = Serial.read();
    if (incomingByte == 'H'){
      digitalWrite(LED, HIGH);
    }
    if (incomingByte == 'L'){
      digitalWrite(LED, LOW);
    }
  }
}
