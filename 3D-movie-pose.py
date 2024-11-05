from mmpose.apis import MMPoseInferencer

img_path = 'movie/10-1.MOV'

inferencer = MMPoseInferencer(pose3d='human3d')

result_generator = inferencer(img_path,  vis_out_dir='result' ,show=True)
result = next(result_generator)
