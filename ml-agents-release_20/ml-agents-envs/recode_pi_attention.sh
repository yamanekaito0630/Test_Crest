a="$1"
r="$2"
v="$3"
e="$4"
t="$5"

mkdir -p log/at${a}/round_im_${r}_robo_slurm_at${a}/render/eval_${e}robo
mkdir -p log/at${a}/v${v}/trial_${t}/eval_${e}robo/render

ffmpeg -r 50 -i log/at${a}/round_im_${r}_robo_slurm_at${a}/attention_gui/img_%d.png -vcodec libx264 -pix_fmt yuv420p -r 50 -s 600x600  log/at${a}/v${v}/trial_${t}/eval_${e}robo/render/attention_movie.mp4

ffmpeg -r 50 -i log/at${a}/round_im_${r}_robo_slurm_at${a}/attention_black_gui/img_%d.png -vcodec libx264 -pix_fmt yuv420p -r 50 -s 600x600  log/at${a}/v${v}/trial_${t}/eval_${e}robo/render/black_attention_movie.mp4
