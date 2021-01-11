"""Makes the videos by combining together all the frames."""
import glob
import os
import subprocess
import enum
import multiprocessing


class Codec(enum.Enum):
    x264 = enum.auto()
    x265 = enum.auto()

codec = Codec.x264
N_WORKERS = 4

base_dir = r"D:\Simulations\CeramicBands\v7\pics"


def do_one_dir(dir_fn: str):
    all_pngs = glob.glob(os.path.join(dir_fn, "*.png"))
    just_fn = os.path.split(dir_fn)[1]
    out_movie = just_fn + f"-{codec.name}.mp4"
    out_movie_full = os.path.join(dir_fn, out_movie)
    has_frames = len(all_pngs) > 10

    has_movie = os.path.isfile(out_movie_full)

    if has_frames and not has_movie:
        print("Got one")
        print(dir_fn)

        os.chdir(dir_fn)

        if codec == Codec.x264:
            codec_args = "-vcodec libx264 -profile:v high444 -refs 16 -crf 0"

        elif codec == Codec.x265:
            codec_args = "-c:v libx265"
            
        else:
            raise ValueError(codec)

        command = fr"C:\Utilities\ffmpeg-20181212-32601fb-win64-static\bin\ffmpeg.exe -f image2 -r 30 -i Case-%04d.png {codec_args} {out_movie}"
        x = subprocess.run(command, capture_output=True)
        
        return str(x)

    else:
        return f"Skipping {dir_fn}"

def produce_dirs():
    all_conts = os.listdir(base_dir)
    all_conts_full = [os.path.join(base_dir, f) for f in all_conts]
    all_dirs = [d for d in all_conts_full if os.path.isdir(d)]

    yield from all_dirs


def do_all_single_process():
    for d in produce_dirs():
        print(do_one_dir(d))


def do_all_multi_process():
    with multiprocessing.Pool(N_WORKERS) as pool:
        for x in pool.imap_unordered(do_one_dir, produce_dirs()):
            print(x)



if __name__ == '__main__':
    do_all_multi_process()