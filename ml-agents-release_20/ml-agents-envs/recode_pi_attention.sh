a="$1"
r="$2"
v="$3"
e="$4"
t="$5"


mkdir -p log/at${a}/${r}robo/v${v}/trial_${t}/eval_${e}robo/render

ffmpeg -y -r 50 -i log/at${a}/${r}robo/v${v}/trial_${t}/eval_${e}robo/attention_gui/img_%d.png -vcodec libx264 -pix_fmt yuv420p -r 50 -s 600x600  log/at${a}/${r}robo/v${v}/trial_${t}/eval_${e}robo/render/attention_movie.mp4

ffmpeg -y -r 50 -i log/at${a}/${r}robo/v${v}/trial_${t}/eval_${e}robo/attention_black_gui/img_%d.png -vcodec libx264 -pix_fmt yuv420p -r 50 -s 600x600  log/at${a}/${r}robo/v${v}/trial_${t}/eval_${e}robo/render/black_attention_movie.mp4
