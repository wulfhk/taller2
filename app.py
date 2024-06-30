import asyncio
import websockets
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from flask import Flask, Response, render_template, jsonify
from flask_bootstrap import Bootstrap
import threading
import os
from ultralytics import YOLO

app = Flask(__name__)
Bootstrap(app)

IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(IMAGE_DIR, exist_ok=True)
IMAGE_PATH = os.path.join(IMAGE_DIR, "image.jpg")

model = YOLO('xddd.pt')

port = int(os.environ.get("PORT", 8080))
websocket_port = int(os.environ.get("WEBSOCKET_PORT", 3001))

latest_image = None
aggression_detected = False
esp32_websocket = None

def is_valid_image(image_bytes):
    try:
        Image.open(BytesIO(image_bytes))
        return True
    except UnidentifiedImageError:
        print("image invalid")
        return False

async def send_alert_to_esp32(message):
    global esp32_websocket
    if esp32_websocket:
        try:
            await esp32_websocket.send(message)
        except websockets.exceptions.ConnectionClosed:
            esp32_websocket = None

async def handle_connection(websocket, path):
    global latest_image, aggression_detected, esp32_websocket
    esp32_websocket = websocket
    while True:
        try:
            message = await websocket.recv()
            print(f"Received message of length: {len(message)}")
            if len(message) > 5000:
                if is_valid_image(message):
                    try:
                        image = Image.open(BytesIO(message))
                        results = model(image, conf=0.7)
                        
                        new_aggression_detected = False
                        for r in results:
                            for box in r.boxes:
                                cls = int(box.cls[0])
                                if cls == 1:  # Ajusta esto según tus necesidades
                                    new_aggression_detected = True
                                    break
                            #if new_aggression_detected:
                             #   break
                            
                            im_array = r.plot()
                            im = Image.fromarray(im_array[..., ::-1])
                            
                            img_byte_arr = BytesIO()
                            im.save(img_byte_arr, format='JPEG')
                            latest_image = img_byte_arr.getvalue()
                        
                        if new_aggression_detected and not aggression_detected:
                            await send_alert_to_esp32("VIOLENCE_DETECTED")
                        elif not new_aggression_detected and aggression_detected:
                            await send_alert_to_esp32("VIOLENCE_CLEARED")
                        
                        aggression_detected = new_aggression_detected
                        
                    except Exception as e:
                        print(f"Error processing image: {e}")
            print()
        except websockets.exceptions.ConnectionClosed:
            break

async def websocket_server():
    server = websocket_port
    await server.wait_closed()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_aggression')
def check_aggression():
    global aggression_detected
    return jsonify({'aggression': aggression_detected})

def get_image():
    global latest_image
    while True:
        if latest_image is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_image + b'\r\n')
        else:
            placeholder_path = os.path.join(IMAGE_DIR, "placeholder.jpg")
            if os.path.exists(placeholder_path):
                with open(placeholder_path, "rb") as f:
                    image_bytes = f.read()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')
            else:
                print("Placeholder image not found")
        # Usa una pequeña pausa para no saturar la CPU
        asyncio.sleep(0.03)  # Ajusta este valor según sea necesario

@app.route('/video_feed')
def video_feed():
    return Response(get_image(), mimetype='multipart/x-mixed-replace; boundary=frame')

async def websocket_handler(websocket, path):
    global latest_image, aggression_detected, esp32_websocket
    esp32_websocket = websocket
    try:
        while True:
            message = await websocket.recv()
            if len(message) > 5000 and is_valid_image(message):
                image = Image.open(BytesIO(message))
                results = model(image, conf=0.7)
                
                new_aggression_detected = any(int(box.cls[0]) == 1 for r in results for box in r.boxes)
                
                for r in results:
                    im_array = r.plot()
                    im = Image.fromarray(im_array[..., ::-1])
                    img_byte_arr = BytesIO()
                    im.save(img_byte_arr, format='JPEG')
                    latest_image = img_byte_arr.getvalue()
                
                if new_aggression_detected != aggression_detected:
                    await send_alert_to_esp32("VIOLENCE_DETECTED" if new_aggression_detected else "VIOLENCE_CLEARED")
                
                aggression_detected = new_aggression_detected
    except websockets.exceptions.ConnectionClosed:
        esp32_websocket = None

async def websocket_server():
    server = await websockets.serve(websocket_handler, '0.0.0.0', websocket_port)
    await server.wait_closed()

def run_flask():
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()
    
    asyncio.get_event_loop().run_until_complete(websocket_server())