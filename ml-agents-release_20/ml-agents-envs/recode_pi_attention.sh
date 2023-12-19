a="$1"
r="$2"
v="$3"

ffmpeg -r 50 -i log/at${a}/round_im_${r}_robo_slurm_at${a}/attention_gui/img_%d.png -vcodec libx264 -pix_fmt yuv420p -r 50 -s 600x600  log/at${a}/round_im_${r}_robo_slurm_at${a}/attention_movie/attention_movie.mp4

ffmpeg -r 50 -i log/at${a}/round_im_${r}_robo_slurm_at${a}/attention_black_gui/img_%d.png -vcodec libx264 -pix_fmt yuv420p -r 50 -s 600x600  log/at${a}/round_im_${r}_robo_slurm_at${a}/attention_black_movie/attention_movie.mp4
