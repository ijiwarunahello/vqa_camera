import cv2
import time

import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

QUESTION = "What does he doing now?"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip_vqa", model_type="vqav2", is_eval=True, device=device
)

def process_frame(frame):
    # ここでフレームごとの処理を実行します。
    # 例えば、BLIPモデルを使ったVQAなど
    image = vis_processors["eval"](frame).unsqueeze(0).to(device)
    question = txt_processors["eval"](QUESTION)
    samples = {"image": image, "text_input": question}

    answer = model.predict_answers(samples=samples, inference_method="generate")
    print(answer)

def main():
    # USBカメラを開く
    cap = cv2.VideoCapture(0)

    # 30fpsにするための待機時間を計算
    wait_time = 1.0 / 30.0

    while True:
        # 現在時刻を取得
        start_time = time.time()

        # フレームを取得
        ret, frame = cap.read()

        if not ret:
            break

        # OpenCVのBGR形式からRGB形式に変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # フレームをPIL.Imageに変換
        pil_image = Image.fromarray(frame_rgb)

        # フレームを処理
        process_frame(pil_image)

        # cv2.imshow("frame", frame)

        # 30fpsになるように待機
        elapsed_time = time.time() - start_time
        if elapsed_time < wait_time:
            time.sleep(wait_time - elapsed_time)

        # 'q'キーが押されたら終了
        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break

    # カメラを閉じる
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
