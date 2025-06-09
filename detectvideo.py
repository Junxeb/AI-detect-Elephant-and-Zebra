# นำเข้า YOLO จากไลบรารี ultralyticsเพื่อให้สามารถสร้างและใช้งานโมเดลได้
from ultralytics import YOLO
import cv2 # ไลบรารีสำหรับจัดการภาพและวิดีโอ

# โหลดโมเดล YOLO โดยใช้ไฟล์โมเดลชื่อ 'best.pt'
model = YOLO('C:/CPE/Ai270/yolov11/best.pt')

# กำหนด path ของวิดีโอ
video_path = 'C:/CPE/Ai270/yolov11/videoplayback.mp4'
cap = cv2.VideoCapture(video_path) # เปิดวิดีโอเพื่ออ่านเฟรมต่อไป


# ตรวจสอบว่าเปิดวิดีโอได้หรือไม่
if not cap.isOpened():
    # หากไม่สามารถเปิดได้ จะพิมพ์ข้อความแจ้งเตือน
    print("ไม่สามารถเปิดวิดีโอได้")
    exit() # หยุดการทำงานของโปรแกรม

# กำหนดขนาดหน้าต่างแสดงผล
window_width = 600
window_height = 400

while cap.isOpened(): #วนลูปอ่านแต่ละเฟรม
    ret, frame = cap.read()
    if not ret: # ในกรณีที่อ่านค่าเฟรมไม่ได้
        # พิมพ์ข้อความแจ้งเตือนหากอ่านค่าเฟรมไม่ได้
        print('ไม่สามารถอ่านเฟรมจากวิดีโอได้')
        break # หยุดการทำงานของลูป

    results = model(frame) # ตรวจจับการรันด้วย YOLO

    anotated_frame = results[0].plot() #แสดงผลลัพธ์
    # แสดงภาพที่มีการตรวจจับแล้วบนหน้าต่างชื่อว่า "Animal Detection"
    cv2.imshow('Animal Detection',anotated_frame)

    # ให้ผู้ใช้กดปุ่ม q เพื่อหยุดการแสดงผลและออกจากลูป
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดการใช้งานกล้องหรือไฟล์วิดีโอ
cap.release()
# ปิดหน้าต่างทั้งหมดที่เปิดโดย OpenCV
cv2.destroyAllWindows()