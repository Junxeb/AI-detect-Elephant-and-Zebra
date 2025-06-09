# นำเข้า YOLO จากไลบรารี ultralyticsเพื่อให้สามารถสร้างและใช้งานโมเดลได้
from ultralytics import YOLO

# โหลดโมเดล YOLO โดยใช้ไฟล์โมเดลชื่อ 'best.pt'
model = YOLO('best.pt')

# นำโมเดลไปใช้กับภาพ test เพื่อทำการตรวจจับวัตถุ
results = model('test/04.jpg')
# แสดงผลลัพธ์ของภาพที่ตรวจจับวัตถุแล้ว
results[0].show()