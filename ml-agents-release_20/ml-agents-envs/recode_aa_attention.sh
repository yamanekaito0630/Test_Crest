a="$1"
r="$2"
v="$3"

ffmpeg -r 50 -i log/aa_at${a}/${r}robo/attention_gui/img_%d.png -vcodec libx264 -pix_fmt yuv420p -r 50 -s 600x600  log/aa_at${a}/${r}robo/attention_movie/attention_movie.mp4

ffmpeg -r 50 -i log/aa_at${a}/${r}robo/attention_black_gui/img_%d.png -vcodec libx264 -pix_fmt yuv420p -r 50 -s 600x600  log/aa_at${a}/${r}robo/attention_black_movie/attention_movie.mp4
