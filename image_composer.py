# from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

import glob
import itertools
import typing
import collections
import contextlib
import os
import multiprocessing
import pathlib

size_4k = (3840, 2160)

class SimulationInfo(typing.NamedTuple):
    throttler_relaxation: float
    adj_strain_ratio: float
    n_steps_minor_max: int
    source_file_name: str
    # scaling: str


    @classmethod
    def from_meta_txt(cls, fn: str) -> "SimulationInfo":
        def line_start(l):
            working = l.strip().split()
            if working:
                return working[0]

        working_dict = {}

        with open(fn) as f_in:
            lines_stripped = (l.strip() for l in f_in)
            lines_to_consider = list(l for l in lines_stripped if line_start(l) in cls.__annotations__)
            for l in lines_to_consider:
                key, val = l.split(maxsplit=1)
                val_as_native_type = (cls.__annotations__[key])(val) 
                working_dict[key] = val_as_native_type

        return cls(**working_dict)


class SubFrame(typing.NamedTuple):
    sim_info: SimulationInfo
    fn: typing.Optional[str]


class Job(typing.NamedTuple):
    sub_frames: typing.List[SubFrame]
    fn_out: str


def make_distinctive_string(this_sim_info: SimulationInfo, all_sim_infos: typing.List[SimulationInfo]) -> str:
    """Find the things which are distinctive and return that in a string"""
    
    # See which ones there are more than one of
    all_vals = collections.defaultdict(set)
    for sim_info in all_sim_infos:
        for key, val in sim_info._asdict().items():
            all_vals[key].add(val)

    not_all_same = {key for key, val_set in all_vals.items() if len(val_set) > 1}

    bits = [f"{key}={val}" for key, val in this_sim_info._asdict().items() if key in not_all_same]
    return "\n".join(bits)


def test_distinctive_strings():
    sim_info_main = SimulationInfo.from_meta_txt(r"E:\Simulations\CeramicBands\v7\pics\70\Meta.txt")
    sim_info_other = SimulationInfo.from_meta_txt(r"E:\Simulations\CeramicBands\v7\pics\71\Meta.txt")

    distinctive_string = make_distinctive_string(sim_info_main, [sim_info_main, sim_info_other])
    assert distinctive_string == "adj_strain_ratio=0.2"


def _tile_dimensions(n: int) -> int:
    """See how many tiled images we need. For example, 4 images fit in 2x2. Seven images need 3x3."""

    current_guess = 1
    while True:
        num_would_fit = current_guess ** 2
        if n <= num_would_fit:
            return current_guess

        current_guess += 1

        if current_guess > 10:
            raise ValueError("What are you crazy???")


T_Num = typing.Union[int, float]
def _add_text_centred_at(image: Image, xy_pos: typing.Tuple[T_Num, T_Num], text: str):
    font = ImageFont.truetype("pala.ttf", 144)

    # Position the text
    text_size = font.getsize(text)

    centred_pos = (int(xy_pos[0]-0.5*text_size[0]), int(xy_pos[1]-0.5*text_size[1]) )
    draw = ImageDraw.Draw(image)
    draw.text(centred_pos, text, font=font)




def compose_images(job: Job):

    all_sim_infos = [sub_frame.sim_info for sub_frame in job.sub_frames]

    with contextlib.ExitStack() as exit_stack:
        images = {}
        for sub_frame in job.sub_frames:
            if sub_frame.fn:
                this_image_file = exit_stack.enter_context(Image.open(sub_frame.fn))
                images[sub_frame] = this_image_file

        dims, modes = set(), set()
        for image in images.values():
            dims.add(image.size)
            modes.add(image.mode)

        if len(dims) != 1:
            raise ValueError(f"Expecting to get a single image dimension. Got {dims}.")

        if len(modes) != 1:
            raise ValueError(f"Expecting a single mode - got {modes}")

        # Compose the images
        dim_single = dims.pop()
        mode = modes.pop()
        n_tiles = _tile_dimensions(len(job.sub_frames))
        dim = tuple(p*n_tiles for p in dim_single)
        big_image = exit_stack.enter_context(Image.new(mode, dim))

        for idx, sub_frame in enumerate(job.sub_frames):
            tile_top, tile_left = divmod(idx, n_tiles)
            tile_bottom = n_tiles - tile_top - 1

            x_offset = tile_left * dim_single[0]
            y_offset = tile_bottom * dim_single[1]

            if sub_frame.fn:
                image = images[sub_frame]
                big_image.paste(image, (x_offset,y_offset))

            # Pop in a caption
            caption = make_distinctive_string(sub_frame.sim_info, all_sim_infos)
            xy_pos = (x_offset + 0.5*dim_single[0], y_offset + 0.9 * dim_single[1])
            _add_text_centred_at(big_image, xy_pos, caption)


        # Don't need it that big
        big_image.thumbnail(size_4k, Image.ANTIALIAS)

        big_image.save(job.fn_out)

    return job.fn_out


class ImageProducer():
    sim_info: SimulationInfo
    d: str
    rel_files: typing.List[str]

    def __init__(self, d: str):
        self.d = d
        self.sim_info = SimulationInfo.from_meta_txt(os.path.join(self.d, "meta.txt"))

        with os.scandir(self.d) as it:
            rel_files = [entry.path for entry in it if entry.name.startswith("Case-")]

        self.rel_files = sorted(rel_files)

    def __iter__(self):
        # Don't include the final file in case it's being worked on or modified.
        return iter(self.rel_files[:-1])


def interleave_directories(dirs: typing.List[str], out_dir: str) -> typing.Iterable[Job]:
    """Zip together the frames, and have it all sorted"""

    image_producers = [ImageProducer(d) for d in dirs]

    def sort_key(image_producer: ImageProducer):
        return image_producer.sim_info

    image_producers.sort(key=sort_key)
    for imp in image_producers:
        print(imp.sim_info, len(imp.rel_files), imp.d)

    for global_frame_idx, fns_or_nones in enumerate(itertools.zip_longest(*image_producers)):
        sub_frames = []
        for image_producer, fn_or_none in zip(image_producers, fns_or_nones):
            sub_frame = SubFrame(sim_info=image_producer.sim_info, fn=fn_or_none)
            sub_frames.append(sub_frame)

        fn_out = os.path.join(out_dir, f"CaseComb-{global_frame_idx:04}.png")
        yield Job(sub_frames=sub_frames, fn_out=fn_out)


def test_compose_image():
    end_dirs = ['70', '71', '72', '73', '74', '75']
    dirs = [os.path.join(r"E:\Simulations\CeramicBands\v7\pics", ed) for ed in end_dirs]

    sub_frames = []
    for d in dirs:
        sub_frame = SubFrame( sim_info=SimulationInfo.from_meta_txt(os.path.join(d, "meta.txt")), fn=os.path.join(d, "Case-0066.png"))        
        sub_frames.append(sub_frame)

    fn_big_test = r"c:\temp\img\composed_image.png"
    sub_frames.sort()

    # None frame means blank image
    temp_frame = sub_frames[1]
    temp_frame = temp_frame._replace(fn=None)
    sub_frames[1] = temp_frame
    job = Job(sub_frames=sub_frames, fn_out=fn_big_test)
    compose_images(job)

    # Make sure we can open it.
    with Image.open(fn_big_test) as f_image:
        assert f_image.size

def test_interleave():
    end_dirs_round2 = ['6K', '6Q', '6R', '6S', '6T', '6U']
    dirs = [os.path.join(r"E:\Simulations\CeramicBands\v7\pics", ed) for ed in end_dirs_round2]
    for job in interleave_directories(dirs, r"c:\temp\img"):
        pass


def make_images():
    end_dirs = ['70', '71', '72', '73', '74', '75', '76', '77', '78']
    dirs = [os.path.join(r"E:\Simulations\CeramicBands\v7\pics", ed) for ed in end_dirs]

    for job in interleave_directories(dirs, r"c:\temp\img"):
        print(job.fn_out)
        compose_images(job)


def do_all_multi_process():
    N_WORKERS=8
    with multiprocessing.Pool(N_WORKERS) as pool:
        end_dirs_round1 = ['70', '71', '72', '73', '74', '75', '76', '77', '78']
        end_dirs_round2 = ['6R', '6S', '6T', '6U']
        end_dirs_round3 = ['6Y', '79', '7A', '7B']
        dirs = [os.path.join(r"E:\Simulations\CeramicBands\v7\pics", ed) for ed in end_dirs_round3]

        for x in pool.imap_unordered(compose_images, interleave_directories(dirs, r"c:\temp\img\round3")):
            print(x)


def print_sim_infos():
    metas = glob.glob(r"E:\Simulations\CeramicBands\v7\pics\*\Meta.txt")
    for fn in sorted(metas):
        dir_end = pathlib.Path(fn)

        try:
            sim_info = SimulationInfo.from_meta_txt(fn)

        except ValueError as e:
            sim_info = None

        print(str(dir_end.parents[0]), sim_info)

if __name__ == "__main__":
    # print_sim_infos()
    do_all_multi_process()

if __name__ == "__ASDASDASD__":
    test_distinctive_strings()
    test_compose_image()

    do_all_multi_process()

    sim_info = SimulationInfo.from_meta_txt(r"E:\Simulations\CeramicBands\v7\pics\70\Meta.txt")
    print(sim_info)

