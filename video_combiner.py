"""Makes the videos by combining together all the frames."""
import glob
import os
import subprocess

base_dir = r"E:\Simulations\CeramicBands\v5\pics"


def do_one_dir(dir_fn: str):
    all_pngs = glob.glob(os.path.join(dir_fn, "*.png"))
    just_fn = os.path.split(dir_fn)[1]
    out_movie = just_fn + ".mp4"
    out_movie_full = os.path.join(dir_fn, out_movie)
    has_frames = len(all_pngs) > 10

    has_movie = os.path.isfile(out_movie_full)

    if has_frames and not has_movie:
        print("Got one")
        print(dir_fn)

        os.chdir(dir_fn)
        command = fr"C:\Utilities\ffmpeg-20181212-32601fb-win64-static\bin\ffmpeg.exe -f image2 -r 30 -i Case-%04d.png -c:v libx265 {out_movie}"
        x = subprocess.run(command)
        print(x)

def do_all():
    all_conts = os.listdir(base_dir)
    all_conts_full = [os.path.join(base_dir, f) for f in all_conts]
    all_dirs = [d for d in all_conts_full if os.path.isdir(d)]
    for d in all_dirs:
        do_one_dir(d)


if __name__ == '__main__':
    do_all()