./build/examples/openpose/openpose.bin --image_dir ~/projects/aDBS_data/videos/calibration_v4/extrinsic_images/extrinsic_images_combined/ \
--3d --3d_views 3 --number_people_max 1 --frame-undistort --logging-level 0 --write_video_fps 30
--write_video_3d ~/projects/aDBS_data/videos/calibration_v4/test_images.avi

./build/examples/openpose/openpose.bin --image_dir ~/projects/aDBS_data/videos/calibration_v4/extrinsic_images/extrinsic_images_combined_renamed/ \
--3d --3d_views 3 --number_people_max 1 --frame-undistort --logging-level 0 --write_video_fps 30 \
--write_video ~/projects/aDBS_data/videos/calibration_v4/test_write_video.avi --display 0

rsync litrtwe@128.32.245.81:~/projects/aDBS_data/videos/calibration_v4/test_write_video.avi ~/Downloads/test_write_video.avi

rsync litrtwe@128.32.245.81:~/projects/openpose/output_images ~/Downloads/output_images -av --dry-run