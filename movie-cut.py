import cv2
import os

def save_all_frames(video_path, dir_path, basename, ext='jpg'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Failed to open video file: {video_path}")
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # frame = cv2.convertScaleAbs(frame, alpha=3, beta=60)
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
            n += 1

    except KeyboardInterrupt:
        print("Process interrupted manually.")

    finally:
        cap.release()

# 自分のジョグ
# input = 'data/movie/light/my-jog.mov'
# output = 'data/image/my-run-light-my-jog-20m'
# output_filename = 'run_light_my_jog_20m'

# 友達の走り
# input = 'data/movie/light/friend-t-run.mov'
# output = 'data/image/my-run-light-friend-t-20m'
# output_filename = 'run_light_friend_t_20m'

# 自分の競技場での走り
# input = 'data/movie/dark/10-1.MOV'
# output = 'data/image/my-run-dark-10'
# output_filename = 'run_dark_10'

# # 自分の競技場での走り2
# input = 'data/movie/dark/10m-3.MOV'
# output = 'data/image/my-run-dark-10-3-lite'
# output_filename = 'run_dark_10-3-lite'

# pro-jでの走り
# input = 'data/movie/pro/pro-j.mov'
# output = 'data/image/pro-j'
# output_filename = 'pro-j'

# オリンピック準決勝3組
input = 'data/movie/olympics/olympics_semifinal_triple_throw.mp4'
output = 'data/image/olympics/olympics_semifinal_triple_throw'
output_filename = 'olympics_semifinal_triple_throw'

save_all_frames(input, output,output_filename)
