from mmpose.apis import MMPoseInferencer

img_path = 'data/image/stand/front.jpg'

# inferencer = MMPoseInferencer(pose3d='human3d')
inferencer = MMPoseInferencer(pose3d='wholebody')


result_generator = inferencer(img_path,  vis_out_dir='result' ,show=True)
result = next(result_generator)